"""Every Fibonna API call the harness makes, for one benchmark project. The counterpart of past_api.py.

One token does everything: it ingests, reads the compile queue and recalls. Only product routes are used:

    GET  /api/v1/projects/resolve                 the benchmark project's id, from its slug
    POST /api/v1/ingest/text                      one turn, as a project memory with its event time
    GET  /api/v1/knowledge/documents/{id}         whether a turn has been chunked and embedded
    GET  /api/v1/brain/compile/jobs               whether the compile queue has drained
    GET  /api/v1/brain/index                      the wiki pages the compile produced
    GET  /api/v1/knowledge/projects/{id}/search   recall

Environment: FIBONNA_API_URL, FIBONNA_API_TOKEN, FIBONNA_BEAM_PROJECT (see `Settings`).
"""

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
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

# How the wait is paced: how often the harness asks, how many turns it checks at once, and how many
# polls in a row must find the compile queue empty (a turn's compile job is queued a moment after
# the turn is chunked, so one empty poll can be that moment).
POLL_SECONDS = 5
TURNS_CHECKED_AT_ONCE = 8
EMPTY_POLLS_TO_SETTLE = 2
PROGRESS_LOG_SECONDS = 30

# The compile job statuses that mean work is still to do, and the one that means it failed for good.
UNFINISHED_JOB_STATUSES = ("pending", "running")
FAILED_JOB_STATUS = "failed"

# One look at what is left to wait for: a description of it, or None when there is nothing left.
Check = Callable[[], Awaitable[str | None]]

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
        return int(stored["document_id"])

    # --- Waiting for compile ------------------------------------------------------------------

    async def wait_until_compiled(
        self, document_ids: list[int], conversation: str | None = None
    ) -> int:
        """Waits until Fibonna has finished with every turn, and returns the project's wiki page count.

        In two steps. First every turn must have been chunked and embedded: only then has its
        compile job been queued. Then the project's compile queue must hold no pending or running
        job, in several polls in a row. A job that failed stops the run.
        """
        started = time.monotonic()
        await self._poll(
            self._all_chunked(document_ids), started, "chunking", conversation
        )
        await self._poll(self._compile_drained(), started, "compiling", conversation)
        pages = len(await self.brain_index())
        log(
            f"compiled into {pages} wiki page(s) after {duration(time.monotonic() - started)}",
            conversation,
        )
        return pages

    async def _poll(
        self, check: Check, started: float, stage: str, conversation: str | None
    ) -> None:
        """Calls `check`, which returns what is left to do or None when it is done, until it is done."""
        last_progress = time.monotonic()
        while (left := await check()) is not None:
            if time.monotonic() - last_progress >= PROGRESS_LOG_SECONDS:
                log(
                    f"{stage}: {left} after {duration(time.monotonic() - started)}",
                    conversation,
                )
                last_progress = time.monotonic()
            await asyncio.sleep(POLL_SECONDS)

    def _all_chunked(self, document_ids: list[int]) -> Check:
        """A check that is done once every turn has chunks. A turn is looked at until it has some."""
        waiting = list(document_ids)

        async def check() -> str | None:
            nonlocal waiting
            slots = asyncio.Semaphore(TURNS_CHECKED_AT_ONCE)

            async def has_chunks(document_id: int) -> bool:
                async with slots:
                    document = await self._call(
                        "GET", f"/api/v1/knowledge/documents/{document_id}"
                    )
                return int(document.get("chunk_count") or 0) > 0

            # Turns are processed in the order they were sent, so the first one that is not ready
            # tells the rest apart from the ones already done: look at no more than a window of them.
            window = waiting[: TURNS_CHECKED_AT_ONCE * 4]
            ready = await asyncio.gather(
                *(has_chunks(document_id) for document_id in window)
            )
            waiting = [d for d, done in zip(window, ready) if not done] + waiting[
                len(window) :
            ]
            return (
                f"{len(waiting)}/{len(document_ids)} turns not yet chunked"
                if waiting
                else None
            )

        return check

    def _compile_drained(self) -> Check:
        """A check that is done once the compile queue has been empty for several polls in a row."""
        empty_polls = 0

        async def check() -> str | None:
            nonlocal empty_polls
            unfinished = sum(
                [await self._compile_jobs(status) for status in UNFINISHED_JOB_STATUSES]
            )
            if await self._compile_jobs(FAILED_JOB_STATUS):
                raise CompileFailed(f"A compile job of project {self.project!r} failed")
            empty_polls = 0 if unfinished else empty_polls + 1
            return (
                None
                if empty_polls >= EMPTY_POLLS_TO_SETTLE
                else f"{unfinished} compile job(s) not finished"
            )

        return check

    async def _compile_jobs(self, status: str) -> int:
        """How many of the project's compile jobs have this status, counted up to one page of 100."""
        found = await self._call(
            "GET",
            "/api/v1/brain/compile/jobs",
            params={"project_id": self.project_id, "status": status, "limit": 100},
        )
        return len(found.get("jobs") or [])

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
