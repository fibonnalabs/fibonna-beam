"""A stand-in for the Fibonna routes the harness uses, served from this process on a local port.

It keeps what the real API keeps for the harness: projects, the turns stored in them with their
event time, a compile queue with one job per turn, and a search that keeps the memories that
happened before a moment (that moment excluded) and echoes the window it applied.

The compile queue works as the real one does. A turn's job is queued some time after the turn is
stored (`hold_compile_enqueue` holds that moment back until `release_compile_enqueue`). The jobs
route lists a project's newest jobs of one status, `limit` at most. A job moves on each time it is
listed: pending, running, then `final_status`. A job that completes leaves a source page for its
turn in the wiki index, as a compile does (`source_pages` False: it leaves none).
"""

import json
import re
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from urllib.parse import parse_qs, urlparse

TOKEN = "test-token"


class LocalServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer.server_bind looks the host name up (getfqdn), which can take half a minute
        # on a machine without a reverse DNS entry; nothing here needs the name.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


class FakeFibonna:
    def __init__(self, projects: dict[str, int]) -> None:
        self.projects = projects  # slug -> id
        self.documents: dict[int, dict] = {}
        self.requests: list[dict] = []
        self.failures: dict[
            str, list[int]
        ] = {}  # "METHOD path" -> statuses to answer, once each
        self.jobs: dict[int, dict] = {}  # compile jobs by id
        self.hold_compile_enqueue = False  # True: stored turns get no compile job yet
        self.instant_drain = False  # True: a job is queued already finished
        self.drains = True  # False: a queued job stays pending
        self.final_status = (
            "completed"  # or "failed" or "skipped": how a job that runs ends
        )
        self.compiled_at_first_search: int | None = None
        # False: a completed compile leaves no page for its turn.
        self.source_pages = True
        # Other pages the wiki holds, by slug.
        self.extra_slugs: list[str] = []
        # Job id -> how many of the next listings, of any status, leave it out.
        self.hidden: dict[int, int] = {}
        self._held: list[int] = []
        self.echo_window = True
        self.keep_event_time = True  # False: dates a turn by when it arrived
        self.pages = 2
        self._lock = threading.Lock()
        self._server = LocalServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def calls(self, method: str, pattern: str) -> list[dict]:
        return [
            r
            for r in self.requests
            if r["method"] == method and re.fullmatch(pattern, r["path"])
        ]

    # --- Routes ---------------------------------------------------------------------------------

    def route(
        self, method: str, path: str, query: dict, body: dict | None
    ) -> tuple[int, object]:
        if method == "GET" and path == "/api/v1/projects/resolve":
            slug = query["slug"]
            if slug not in self.projects:
                return 404, {"detail": "Project not found"}
            return 200, {
                "id": self.projects[slug],
                "slug": slug,
                "name": slug,
                "repositories": [],
                "matched_by": "slug",
            }
        if method == "POST" and path == "/api/v1/ingest/text":
            return self.store(body)
        if method == "GET" and path == "/api/v1/brain/compile/jobs":
            return self.list_compile_jobs(query)
        if method == "GET" and path == "/api/v1/brain/index":
            return 200, self.wiki_pages(int(query["project_id"]))
        if method == "GET" and (
            m := re.fullmatch(r"/api/v1/knowledge/projects/(\d+)/search", path)
        ):
            return self.search(int(m[1]), query)
        return 404, {"detail": "Not Found"}

    def store(self, body: dict) -> tuple[int, object]:
        if (
            body["kb_type"] != "project"
            or body["project_id"] not in self.projects.values()
        ):
            return 404, {"detail": "Project not found"}
        markdown = f"# {body['title']}\n\n{body['content']}"
        for document_id, document in self.documents.items():
            if (
                document["markdown"] == markdown
                and document["project_id"] == body["project_id"]
            ):
                return 200, {
                    "status": "duplicate",
                    "document_id": document_id,
                    "occurred_at": document["occurred_at"],
                    "job_ids": [],
                }
        document_id = len(self.documents) + 1
        occurred_at = utc(
            body["occurred_at"] if self.keep_event_time else "2030-01-01T00:00:00Z"
        ).isoformat()
        self.documents[document_id] = {
            "project_id": body["project_id"],
            "markdown": markdown,
            "content": body["content"],
            "occurred_at": occurred_at,
        }
        if self.hold_compile_enqueue:
            self._held.append(document_id)
        else:
            self.enqueue_compile(document_id)
        return 200, {
            "status": "queued",
            "document_id": document_id,
            "occurred_at": occurred_at,
            "occurred_precision": body["occurred_precision"],
            "job_ids": [document_id],
        }

    def enqueue_compile(self, document_id: int) -> None:
        job_id = len(self.jobs) + 1
        self.jobs[job_id] = {
            "id": job_id,
            "project_id": self.documents[document_id]["project_id"],
            "source_type": "document",
            "source_id": str(document_id),
            "status": self.final_status if self.instant_drain else "pending",
            "error": "boom" if self.final_status == "failed" else None,
        }

    def release_compile_enqueue(self) -> None:
        """The moment the stored turns' compile jobs are queued."""
        with self._lock:
            self.hold_compile_enqueue = False
            for document_id in self._held:
                self.enqueue_compile(document_id)
            self._held.clear()

    def wiki_pages(self, project_id: int) -> list[dict]:
        """The project's pages: `pages` of other kinds, and a source page per turn compiled."""
        pages = [
            {
                "slug": f"page-{i}",
                "title": f"Page {i}",
                "page_type": "concept",
                "one_liner": "x",
            }
            for i in range(self.pages)
        ]
        pages += [
            {"slug": slug, "title": slug, "page_type": "concept", "one_liner": "x"}
            for slug in self.extra_slugs
        ]
        if self.source_pages:
            for job in self.jobs.values():
                if (
                    job["project_id"] == project_id
                    and job["source_type"] == "document"
                    and job["status"] == "completed"
                ):
                    slug = f"src-document-{job['source_id']}-quick-note-turn-md"
                    pages.append(
                        {
                            "slug": slug,
                            "title": "Turn",
                            "page_type": "source_summary",
                            "one_liner": "x",
                        }
                    )
        return pages

    def list_compile_jobs(self, query: dict) -> tuple[int, object]:
        limit = int(query.get("limit", 20))
        if not 1 <= limit <= 100:
            return 422, {"detail": "limit must be between 1 and 100"}
        project_id = int(query["project_id"])
        status = query.get("status")
        newest_first = sorted(self.jobs.values(), key=lambda job: -job["id"])
        omitted = {job_id for job_id, asks in self.hidden.items() if asks > 0}
        self.hidden = {job_id: max(asks - 1, 0) for job_id, asks in self.hidden.items()}
        listed = [
            job
            for job in newest_first
            if job["project_id"] == project_id
            and job["id"] not in omitted
            and (status is None or job["status"] == status)
        ][:limit]
        views = [dict(job) for job in listed]
        for job in listed:
            self.move_on(job)
        return 200, {"jobs": views}

    def move_on(self, job: dict) -> None:
        if not self.drains:
            return
        if job["status"] == "pending":
            job["status"] = "running"
        elif job["status"] == "running":
            job["status"] = self.final_status
            job["error"] = "boom" if self.final_status == "failed" else None

    def compiled_jobs(self) -> int:
        return sum(job["status"] == "completed" for job in self.jobs.values())

    def search(self, project_id: int, query: dict) -> tuple[int, object]:
        if self.compiled_at_first_search is None:
            self.compiled_at_first_search = self.compiled_jobs()
        before = utc(query["occurred_before"])
        words = set(re.findall(r"\w+", query["q"].lower()))
        found = []
        for document_id, document in self.documents.items():
            if (
                document["project_id"] != project_id
                or utc(document["occurred_at"]) >= before
            ):
                continue
            overlap = len(words & set(re.findall(r"\w+", document["content"].lower())))
            if overlap:
                found.append((overlap / len(words), document_id, document))
        found.sort(key=lambda hit: -hit[0])
        response = {
            "results": [
                {
                    "id": document_id,
                    "content": document["markdown"],
                    "similarity": similarity,
                    "source": "chunk",
                    "metadata": {"occurred_at": document["occurred_at"]},
                }
                for similarity, document_id, document in found[: int(query["top_k"])]
            ],
            "total": len(found),
        }
        if self.echo_window:
            response["time_window"] = {
                "occurred_after": None,
                "occurred_before": before.isoformat(),
            }
        return 200, response

    # --- HTTP -----------------------------------------------------------------------------------

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # keep the test output clean
                pass

            def handle_request(self) -> None:
                url = urlparse(self.path)
                query = {key: values[0] for key, values in parse_qs(url.query).items()}
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                with fake._lock:
                    fake.requests.append(
                        {
                            "method": self.command,
                            "path": url.path,
                            "query": query,
                            "body": body,
                            "headers": dict(self.headers),
                        }
                    )
                    if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                        status, payload = 401, {"detail": "Not authenticated"}
                    elif fake.failures.get(f"{self.command} {url.path}"):
                        status = fake.failures[f"{self.command} {url.path}"].pop(0)
                        payload = {"detail": "failure"}
                    else:
                        status, payload = fake.route(
                            self.command, url.path, query, body
                        )
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = handle_request

        return Handler
