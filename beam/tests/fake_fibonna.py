"""A stand-in for the Fibonna routes the harness uses, served from this process on a local port.

It keeps what the real API keeps for the harness: projects, the turns stored in them with their
event time, a compile queue that drains over a few polls, and a search that keeps the memories that
happened before a moment (that moment excluded) and echoes the window it applied.
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
        self.chunk_polls_before_ready = 1
        self.pending_polls_left = 3  # polls of the compile queue that still find a job
        self.failed_compile_jobs = 0
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
        if method == "GET" and (
            m := re.fullmatch(r"/api/v1/knowledge/documents/(\d+)", path)
        ):
            return self.read_document(int(m[1]))
        if method == "GET" and path == "/api/v1/brain/compile/jobs":
            return 200, {"jobs": self.compile_jobs(query["status"])}
        if method == "GET" and path == "/api/v1/brain/index":
            return 200, [
                {
                    "slug": f"page-{i}",
                    "title": f"Page {i}",
                    "page_type": "concept",
                    "one_liner": "x",
                }
                for i in range(self.pages)
            ]
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
            "chunk_polls": 0,
        }
        return 200, {
            "status": "queued",
            "document_id": document_id,
            "occurred_at": occurred_at,
            "occurred_precision": body["occurred_precision"],
            "job_ids": [document_id],
        }

    def read_document(self, document_id: int) -> tuple[int, object]:
        document = self.documents.get(document_id)
        if document is None:
            return 404, {"detail": "Document not found"}
        document["chunk_polls"] += 1
        ready = document["chunk_polls"] > self.chunk_polls_before_ready
        return 200, {
            "document_id": document_id,
            "chunk_count": 1 if ready else 0,
            "content": document["content"],
        }

    def compile_jobs(self, status: str) -> list[dict]:
        if status == "pending" and self.pending_polls_left > 0:
            self.pending_polls_left -= 1
            return [{"id": 1, "status": "pending"}]
        if status == "failed" and self.failed_compile_jobs:
            return [{"id": 2, "status": "failed"}]
        return []

    def search(self, project_id: int, query: dict) -> tuple[int, object]:
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
