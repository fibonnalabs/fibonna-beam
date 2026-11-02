"""Every Fibonna API call the harness makes, for one benchmark project. The counterpart of past_api.py.

One token does everything: it ingests, reads the compile queue and recalls. Only product routes are used:

    GET  /api/v1/projects/resolve                 the benchmark project's id, from its slug
    POST /api/v1/ingest/text                      one turn, as a project memory with its event time
    GET  /api/v1/brain/compile/jobs               each turn's compile job: queued, and finished
    GET  /api/v1/brain/index                      the wiki pages the compile produced, and which turns it did
    GET  /api/v1/knowledge/projects/{id}/search   recall

Environment: FIBONNA_API_URL, FIBONNA_API_TOKEN, FIBONNA_BEAM_PROJECT (see `Settings`).
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx

from log import duration, log
from past_api import PastApiError

# One recall: at most 100 results within 8,000 tokens of evidence, the budget past.dev's run uses.
RECALL_LIMIT = 100
RECALL_MAX_TOKENS = 8_000

# A rate limit, a server error or a dropped connection is retried: four attempts in all.
RETRY_DELAYS_SECONDS = (1, 2, 4)
TIMEOUT_SECONDS = 300

# The WAF in front of the API blocks the default httpx User-Agent; a product tag in a browser-style
# string passes it (the same tag fibonna-mcp sends).
USER_AGENT = (
    "Mozilla/5.0 (fibonna-beam/0.1; +https://github.com/fibonnalabs/fibonna-beam)"
)

# How the wait is paced: how often the harness asks, how long it waits while nothing moves before it
# gives up, and how often it says where it is. A job that stalls while running is taken back after 30
# minutes and then waits for the next drain tick, so a stop any sooner than an hour would stop a run
# that is about to recover.
POLL_SECONDS = 5
STALL_TIMEOUT_SECONDS = 60 * 60
PROGRESS_LOG_SECONDS = 30

# A compile job's statuses. The jobs route lists a project's newest jobs of one status, 100 at most
# and with no way to page on, and it orders them by when they were created, not when they finished.
# So the harness asks for each status in turn and keeps what it has seen, and a job it has seen
# queued that no listing shows any more is settled by `_settle_jobs_out_of_sight`. It asks again
# every OBSERVE_EVERY_TURNS turns while sending, so that a queue that drains fast cannot push the
# first turns' jobs out of sight before the wait begins.
COMPILE_JOB_STATUSES = ("pending", "running", "completed", "failed", "skipped")
COMPILE_JOBS_LIMIT = 100
OBSERVE_EVERY_TURNS = 25
# A job missing from the listings this many rounds in a row has left the queue (see above).
ROUNDS_BEFORE_A_JOB_IS_SETTLED = 2

# A finished compile writes one source page for the document it compiled, at src-document-<id>-<title>.
# For a turn whose job cannot be seen, that page says the turn was compiled.
SOURCE_PAGE_SLUG = re.compile(r"src-document-(\d+)(?:-|$)")

REQUIRED_ENVIRONMENT = ("FIBONNA_API_URL", "FIBONNA_API_TOKEN", "FIBONNA_BEAM_PROJECT")
CONVERSATION_PLACEHOLDER = "{conversation}"


class FibonnaApiError(PastApiError):
    """A call Fibonna refused. It is a PastApiError so that run.answer_question, which the Fibonna
    runner shares with the Past one, scores a recall that failed even after retries 0 and goes on."""


class ProjectMissing(RuntimeError):
    """The benchmark project does not exist. The harness never creates one."""


class CompileFailed(RuntimeError):
    """Fibonna could not compile a turn. The run stops rather than score a conversation whose
    memory is incomplete."""


class CompileTimedOut(CompileFailed):
    """Compiling stopped moving: turns still have no compile job, or their jobs are not finishing."""


class StallClock:
    """When anything last moved, shared by the conversations that wait at the same time.

    The stack's compile queue is one queue for every project, and it takes the newest jobs first.
    While a backlog drains, the jobs of the conversations sent first sit untouched for as long as
    the later ones take. A conversation is stalled when nothing has moved in any of them, not when
    only its own jobs have stood still.
    """

    def __init__(self) -> None:
        self._last_moved = time.monotonic()

    def moved(self) -> None:
        self._last_moved = time.monotonic()

    def still_for(self) -> float:
        return time.monotonic() - self._last_moved


@dataclass(frozen=True)
class CompileJob:
    """A turn's compile job as the jobs route reports it."""

    id: int
    status: str
    error: str | None


@dataclass(frozen=True)
class CompileProgress:
    """Where the compile of a set of turns stands. Equal progress means nothing has moved."""

    without_job: tuple[int, ...]  # the documents no compile job has been queued for yet
    pending: int
    running: int
    compiled: int

    @property
    def finished(self) -> bool:
        return not self.without_job and not self.pending and not self.running

    def describe(self) -> str:
        text = (
            f"{self.compiled} compiled, {self.pending} pending, {self.running} running"
        )
        if self.without_job:
            first = ", ".join(str(d) for d in self.without_job[:5])
            text += f", {len(self.without_job)} with no compile job yet (documents {first}…)"
        return text


@dataclass(frozen=True)
class Settings:
    """Where the run goes: the API, the token, and the benchmark project's slug.

    The slug may hold `{conversation}`, which is replaced by each conversation's id, so that every
    conversation gets a project of its own, as past.dev's harness gives it. Without it, all the
    conversations of the run share the one project, and a recall would find the others' turns.
    """

    base_url: str
    token: str
    project: str

    @classmethod
    def from_environment(cls) -> "Settings":
        missing = [
            name
            for name in REQUIRED_ENVIRONMENT
            if not os.environ.get(name, "").strip()
        ]
        if missing:
            raise SystemExit(
                f"Set {', '.join(missing)} before running (see README.md)."
            )
        return cls(
            base_url=os.environ["FIBONNA_API_URL"].strip(),
            token=os.environ["FIBONNA_API_TOKEN"].strip(),
            project=os.environ["FIBONNA_BEAM_PROJECT"].strip(),
        )

    @property
    def one_project_per_conversation(self) -> bool:
        return CONVERSATION_PLACEHOLDER in self.project

    def project_for(self, conversation_id: str) -> str:
        return self.project.replace(CONVERSATION_PLACEHOLDER, conversation_id)


class FibonnaApi:
    """Fibonna, for one project: ingest its turns, wait until compile has finished, recall."""

    def __init__(self, base_url: str, token: str, project: str) -> None:
        self.project = project  # the slug
        self._project_id: int | None = None
        self._compile_jobs: dict[int, CompileJob] = {}  # by document id
        # For each document with a queued job, the rounds in a row no listing has shown it.
        self._rounds_out_of_sight: dict[int, int] = {}
        # The documents the wiki holds a source page for, whatever their job.
        self._compiled_documents: set[int] = set()
        self._turns_sent = 0
        self._a_listing_was_full = False
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {token.strip()}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout=TIMEOUT_SECONDS,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )

    async def close(self) -> None:
        await self._http.aclose()

    # --- The project --------------------------------------------------------------------------

    async def open_project(self) -> int:
        """The project's id. It must exist already: the harness never creates projects."""
        try:
            found = await self._call(
                "GET", "/api/v1/projects/resolve", params={"slug": self.project}
            )
        except FibonnaApiError as error:
            if error.status == 404:
                raise ProjectMissing(
                    f"Fibonna has no project {self.project!r}. Create it first (see README.md)."
                ) from error
            raise
        self._project_id = int(found["id"])
        return self._project_id

    @property
    def project_id(self) -> int:
        if self._project_id is None:
            raise RuntimeError("open_project() comes first")
        return self._project_id

    # --- Ingesting ----------------------------------------------------------------------------

    async def ingest_turn(self, item: dict) -> int:
        """Stores one turn, an item as dataset.to_ingest_item makes it, and returns its document id.

        The turn is stored with its own time as its event time, so recall can answer as of a past
        moment. Its title carries the author and the turn's id: Fibonna stores identical text once,
        and two turns that say the same ("Thanks!") must stay two memories. Sending a turn again
        (a retry whose answer was lost) finds the memory the first attempt stored.
        """
        author = item["metadata"]["author"]
        stored = await self._call(
            "POST",
            "/api/v1/ingest/text",
            {
                "content": item["content"],
                "title": f"{author} ({item['metadata']['role']}) {item['id']}",
                "kb_type": "project",
                "project_id": self.project_id,
                "category": "note",
                "occurred_at": item["timestamp"],
                "occurred_precision": "exact",
                "metadata": item["metadata"],
            },
        )
        if stored.get("status") not in ("queued", "duplicate"):
            raise FibonnaApiError(f"Fibonna did not take turn {item['id']}: {stored}")
        # A turn whose event time Fibonna did not keep would be dated by when it arrived.
        kept = stored.get("occurred_at")
        if kept is None or moment(kept) != moment(item["timestamp"]):
            raise FibonnaApiError(
                f"Fibonna kept the event time {kept!r} for turn {item['id']}, sent {item['timestamp']}"
            )
        self._turns_sent += 1
        if self._turns_sent % OBSERVE_EVERY_TURNS == 0:
            await self._observe_compile_jobs()
        return int(stored["document_id"])

    # --- Waiting for compile ------------------------------------------------------------------

    async def wait_until_compiled(
        self,
        document_ids: list[int],
        conversation: str | None = None,
        stall_timeout: float = STALL_TIMEOUT_SECONDS,
        clock: StallClock | None = None,
    ) -> int:
        """Waits until every turn's own compile job has completed, and returns the wiki page count.

        A turn is compiled when the compile queue says so, for that turn: a compile job for its
        document exists and has completed (or, for a job the queue no longer shows, the wiki holds
        the turn's source page). Not when it has been chunked, and not when the queue looks empty: a
        job is queued a while after the turn is stored, and until then the queue has nothing to say
        about it. A job that failed or was skipped stops the run. So does a wait in which nothing
        moves for `stall_timeout` seconds: no job is queued for a turn, or a queued job never
        finishes. Waits that share a `clock` move it for each other. The wait has no other limit; a
        large conversation takes hours.
        """
        started = last_logged = time.monotonic()
        clock = clock or StallClock()
        previous: CompileProgress | None = None
        while True:
            await self._observe_compile_jobs()
            progress = await self._compile_progress(document_ids)
            if progress.finished:
                break
            now = time.monotonic()
            if progress != previous:
                previous = progress
                clock.moved()
            elif clock.still_for() >= stall_timeout:
                raise CompileTimedOut(
                    f"Compiling project {self.project!r} has not moved for {duration(stall_timeout)}: "
                    f"{progress.describe()}. A turn with no compile job was never queued for compile "
                    "(is compile on in the stack, and did the turn get chunked?); a job that stays "
                    "pending or running is not being drained (is the compile drain running, and the stack's "
                    "monthly model spend cap not reached?)."
                    + (
                        f" The jobs route lists only the newest {COMPILE_JOBS_LIMIT} jobs of a status, "
                        "so a job completed long ago may be out of sight, and the wiki holds no source "
                        "page for these turns either (see README.md, 'Seeing the jobs')."
                        if self._a_listing_was_full
                        else ""
                    )
                )
            if now - last_logged >= PROGRESS_LOG_SECONDS:
                log(
                    f"compiling: {progress.describe()} after {duration(now - started)}",
                    conversation,
                )
                last_logged = now
            await asyncio.sleep(POLL_SECONDS)
        pages = len(await self.brain_index())
        log(
            f"compiled into {pages} wiki page(s) after {duration(time.monotonic() - started)}",
            conversation,
        )
        return pages

    async def pages_of_a_compiled_project(self) -> int:
        """The wiki page count of a project an earlier run has sent and waited for.

        For `--reuse`, which asks again without sending. What the compile queue still lists must be
        finished and complete, and the project must hold wiki pages. It cannot show that every turn
        was compiled: jobs that finished long ago are no longer listed (see COMPILE_JOB_STATUSES).
        """
        await self._observe_compile_jobs()
        unfinished = {
            document_id: job
            for document_id, job in self._compile_jobs.items()
            if job.status != "completed"
        }
        if unfinished:
            document_id, job = next(iter(unfinished.items()))
            raise CompileFailed(
                f"Project {self.project!r} is not compiled: {len(unfinished)} compile job(s) are not "
                f"completed, such as job {job.id} of document {document_id}, which is {job.status}."
            )
        pages = len(await self.brain_index())
        if not pages:
            raise CompileFailed(
                f"Project {self.project!r} has no wiki pages: an earlier run must send its turns "
                "and wait for them to compile before --reuse can ask about them."
            )
        return pages

    async def _observe_compile_jobs(self) -> None:
        """Asks for the project's compile jobs of every status and keeps each document's newest."""
        # Each document's newest job in this round, and the statuses whose listing hit the limit
        # and so may hide jobs.
        listed: dict[int, int] = {}
        full: set[str] = set()
        for status in COMPILE_JOB_STATUSES:
            found = await self._call(
                "GET",
                "/api/v1/brain/compile/jobs",
                params={
                    "project_id": self.project_id,
                    "status": status,
                    "limit": COMPILE_JOBS_LIMIT,
                },
            )
            if len(found.get("jobs") or []) >= COMPILE_JOBS_LIMIT:
                full.add(status)
                self._a_listing_was_full = True
            for job in found.get("jobs") or []:
                if job.get("source_type") != "document":
                    continue
                document_id = int(job["source_id"])
                listed[document_id] = max(listed.get(document_id, 0), job["id"])
                known = self._compile_jobs.get(document_id)
                if known is None or job["id"] >= known.id:
                    self._compile_jobs[document_id] = CompileJob(
                        id=job["id"], status=job["status"], error=job.get("error")
                    )
        self._settle_jobs_out_of_sight(listed, full)

    def _settle_jobs_out_of_sight(self, listed: dict[int, int], full: set[str]) -> None:
        """Settles the jobs last seen queued or running that no listing shows now: they completed.

        The jobs route orders by when a job was created, not when it finished. A job that is retried
        finishes after newer ones, and so lands already out of the newest completed jobs; one that
        finishes while the queue is not being asked about drops out the same way. Kept as pending,
        it would hold the wait until the stall timeout although it compiled. A job missing from the
        pending, running, failed and skipped listings, when none of those is full and so each shows
        all it holds, is neither queued nor failed, so it completed. A retry takes a job from running
        back to pending between two of the asks of one round, so it must be missing from
        ROUNDS_BEFORE_A_JOB_IS_SETTLED rounds in a row.
        """
        everything_else_shown = not full & {"pending", "running", "failed", "skipped"}
        for document_id, job in list(self._compile_jobs.items()):
            if (
                job.status not in ("pending", "running")
                or listed.get(document_id) == job.id
                or not everything_else_shown
            ):
                self._rounds_out_of_sight.pop(document_id, None)
                continue
            rounds = self._rounds_out_of_sight.get(document_id, 0) + 1
            if rounds < ROUNDS_BEFORE_A_JOB_IS_SETTLED:
                self._rounds_out_of_sight[document_id] = rounds
                continue
            self._rounds_out_of_sight.pop(document_id, None)
            self._compile_jobs[document_id] = CompileJob(
                id=job.id, status="completed", error=None
            )

    async def _compile_progress(self, document_ids: list[int]) -> CompileProgress:
        """Where each document's compile stands, as last observed. A failed job raises.

        A document whose job has not been seen is looked up in the wiki: a finished compile wrote its
        source page, which a project compiled in full keeps however many jobs the route has dropped.
        """
        unplaced = [
            document_id
            for document_id in document_ids
            if document_id not in self._compile_jobs
            and document_id not in self._compiled_documents
        ]
        if unplaced:
            self._compiled_documents |= await self._documents_with_a_source_page()
        without_job: list[int] = []
        counts = {"pending": 0, "running": 0, "completed": 0}
        for document_id in document_ids:
            job = self._compile_jobs.get(document_id)
            if job is None:
                if document_id in self._compiled_documents:
                    counts["completed"] += 1
                else:
                    without_job.append(document_id)
            elif job.status in counts:
                counts[job.status] += 1
            else:
                raise CompileFailed(
                    f"Fibonna did not compile document {document_id} of project {self.project!r}: "
                    f"its compile job {job.id} is {job.status}"
                    + (f" ({job.error})" if job.error else "")
                )
        return CompileProgress(
            without_job=tuple(without_job),
            pending=counts["pending"],
            running=counts["running"],
            compiled=counts["completed"],
        )

    async def _documents_with_a_source_page(self) -> set[int]:
        """The documents the project's wiki holds a source page for."""
        return {
            int(match[1])
            for page in await self.brain_index()
            if (match := SOURCE_PAGE_SLUG.match(page.get("slug") or ""))
        }

    async def brain_index(self) -> list[dict]:
        """The wiki pages of the project: `slug`, `title`, `page_type` and a one-line summary each."""
        return await self._call(
            "GET", "/api/v1/brain/index", params={"project_id": self.project_id}
        )

    # --- Recalling ----------------------------------------------------------------------------

    async def recall(self, query: str, as_of: str, identity: str | None = None) -> dict:
        """What the project remembers about `query`, as it stood at the moment `as_of`.

        The answer model reads this dict as returned: `results` holds the evidence, each with its
        `text`. `identity` is part of the call the shared runner makes; a Fibonna token is one user,
        so it is not sent.
        """
        # Search keeps the memories that happened before a moment, that moment excluded; the turn
        # at `as_of` itself, the update a knowledge_update question asks after, must stay in.
        before = (moment(as_of) + timedelta(microseconds=1)).isoformat()
        found = await self._call(
            "GET",
            f"/api/v1/knowledge/projects/{self.project_id}/search",
            params={
                "q": query,
                "top_k": RECALL_LIMIT,
                "max_tokens": RECALL_MAX_TOKENS,
                "occurred_before": before,
                "include_personal": False,
                "include_repo_ingest": False,
            },
        )
        # Fibonna echoes the window it applied; one that is missing means it ignored ours.
        echoed = (found.get("time_window") or {}).get("occurred_before")
        if echoed is None or moment(echoed) != moment(before):
            raise FibonnaApiError(
                f"Search did not apply the moment {as_of}: it echoed {echoed!r}"
            )
        return {
            "asOf": as_of,
            "total": found.get("total", 0),
            "results": [
                {
                    "text": result["content"],
                    "occurredAt": (result.get("metadata") or {}).get("occurred_at"),
                    "source": result.get("source"),
                    "similarity": result.get("similarity"),
                }
                for result in found.get("results") or []
            ],
        }

    # --- Transport ----------------------------------------------------------------------------

    async def _call(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        params: dict | None = None,
    ) -> Any:
        attempts = len(RETRY_DELAYS_SECONDS) + 1
        for attempt in range(attempts):
            is_last_attempt = attempt == attempts - 1
            try:
                response = await self._http.request(
                    method, path, json=body, params=params
                )
            except httpx.HTTPError as error:
                if is_last_attempt:
                    raise
                log(
                    f"{method} {path} failed ({type(error).__name__}); retrying in {RETRY_DELAYS_SECONDS[attempt]}s"
                )
                await asyncio.sleep(RETRY_DELAYS_SECONDS[attempt])
                continue
            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and not is_last_attempt:
                log(
                    f"{method} {path} returned {response.status_code}; retrying in {RETRY_DELAYS_SECONDS[attempt]}s"
                )
                await asyncio.sleep(RETRY_DELAYS_SECONDS[attempt])
                continue
            if response.is_error:
                raise FibonnaApiError(
                    f"{method} {path} returned {response.status_code}: {response.text[:1_000]}",
                    status=response.status_code,
                )
            return response.json() if response.content else {}
        raise AssertionError("unreachable")


def moment(text: str) -> datetime:
    """A moment as Fibonna or the dataset writes it, ending in Z or in an offset."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
