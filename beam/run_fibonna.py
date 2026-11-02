"""Run BEAM against Fibonna's API.

    python run_fibonna.py --split 100k

The run has the same three phases as run.py, each over every conversation:

    1. send: ingest every turn into the conversation's project, with its own time as its event time
    2. wait until every turn's compile job has completed
    3. score: ask each question, answer it from what Fibonna recalls, and score the answer

The dataset, ExaBase's prompts and the scoring are those of run.py; so are the answer step and the
summary's layout. Only where the memory lives differs, and `fibonna_api.py` is all of that.

With --reuse, a run asks again about conversations an earlier run sent and waited for: it sends
nothing and waits for nothing, and stops if a project is still compiling or has no wiki pages.
That is how to score the same memory with another judge.

Environment: FIBONNA_API_URL, FIBONNA_API_TOKEN, FIBONNA_BEAM_PROJECT, OPENROUTER_API_KEY. To judge
with a second model, of another family, BEAM_JUDGE_MODEL.
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import exabase_prompts
import llm
import scoring
from dataset import (
    SPLITS,
    Conversation,
    build_conversation,
    load_rows,
    to_ingest_item,
    utc_iso,
)
from fibonna_api import (
    RECALL_LIMIT,
    RECALL_MAX_TOKENS,
    STALL_TIMEOUT_SECONDS,
    CompileFailed,
    FibonnaApi,
    ProjectMissing,
    Settings,
    StallClock,
)
from log import duration, log
from run import (
    CONVERSATIONS_SCORED_AT_ONCE,
    CONVERSATIONS_SENT_AT_ONCE,
    HARNESS_VERSION,
    PROGRESS_LOG_SECONDS,
    RECALLS_AT_ONCE,
    answer_question,
    positive_number,
    print_summary,
)

REQUIRED_ENVIRONMENT = ("OPENROUTER_API_KEY",)


# --- Phase 1, send: every turn ---------------------------------------------------------------------


@dataclass
class Sent:
    """A conversation whose turns are all in Fibonna, with the project that holds them."""

    conversation: Conversation
    api: FibonnaApi
    document_ids: list[int]


async def ingest(api: FibonnaApi, conversation: Conversation) -> list[int]:
    """Stores every turn, one after another, and returns the document id of each.

    One at a time keeps the conversation's order. Other conversations send meanwhile.
    """
    items = [to_ingest_item(conversation, message) for message in conversation.messages]
    log(f"ingesting {len(items)} turns", conversation.id)
    document_ids = []
    for item in items:
        document_ids.append(await api.ingest_turn(item))
        if len(document_ids) % 100 == 0 or len(document_ids) == len(items):
            log(f"sent {len(document_ids)}/{len(items)} turns", conversation.id)
    return document_ids


# --- The run -----------------------------------------------------------------------------------


class Run:
    """What every conversation of one run shares."""

    def __init__(self, args: argparse.Namespace, settings: Settings) -> None:
        self.split = args.split
        self.settings = settings
        self.output = Path(args.output or f"results/fibonna-{args.split}")
        self.stall_timeout = args.compile_stall_minutes * 60
        self.reusing = args.reuse
        self.results = self.output / "results.jsonl"
        self.sending_slots = asyncio.Semaphore(CONVERSATIONS_SENT_AT_ONCE)
        self.scoring_slots = asyncio.Semaphore(CONVERSATIONS_SCORED_AT_ONCE)
        self.recall_slots = asyncio.Semaphore(RECALLS_AT_ONCE)
        self.write_lock = asyncio.Lock()
        self.apis: list[FibonnaApi] = []  # closed when the run ends, however it ends
        # Every conversation's wait moves it, so that one whose jobs the queue reaches last is not
        # stopped while the others compile.
        self.stall_clock = StallClock()
        self.compiled_pages: dict[str, int] = {}
        self.turns_sent = 0
        self.processed = 0
        self.scored = 0

    async def send(self, row: dict) -> Sent:
        async with self.sending_slots:
            conversation = build_conversation(row)
            log(
                f"{'reusing' if self.reusing else 'sending'}: {len(conversation.messages)} turns, "
                f"{len(conversation.questions)} questions",
                conversation.id,
            )
            api = FibonnaApi(
                self.settings.base_url,
                self.settings.token,
                self.settings.project_for(conversation.id),
            )
            self.apis.append(api)
            project_id = await api.open_project()
            log(f"project {api.project} is {project_id}", conversation.id)
            document_ids = [] if self.reusing else await ingest(api, conversation)
            self.turns_sent += len(document_ids)
            return Sent(conversation, api, document_ids)

    async def wait(self, sent: Sent, started: float) -> None:
        if self.reusing:
            pages = await sent.api.pages_of_a_compiled_project()
        else:
            pages = await sent.api.wait_until_compiled(
                sent.document_ids,
                sent.conversation.id,
                self.stall_timeout,
                self.stall_clock,
            )
        self.compiled_pages[sent.conversation.id] = pages
        self.processed += 1
        log(
            f"processed after {duration(time.monotonic() - started)}",
            sent.conversation.id,
        )

    async def log_processing_progress(self, total: int, started: float) -> None:
        """A progress line every PROGRESS_LOG_SECONDS, until it is cancelled."""
        while True:
            await asyncio.sleep(PROGRESS_LOG_SECONDS)
            log(
                f"processing: {self.processed}/{total} conversations done after "
                f"{duration(time.monotonic() - started)}"
            )

    async def score(self, sent: Sent, total: int) -> None:
        async with self.scoring_slots:
            conversation = sent.conversation
            started = time.monotonic()
            rows = await asyncio.gather(
                *(
                    answer_question(sent.api, conversation, question, self.recall_slots)
                    for question in conversation.questions
                )
            )
            await self.save(rows)
            self.scored += 1
            mean = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
            log(
                f"scored: mean {mean:.3f} over {len(rows)} questions in {duration(time.monotonic() - started)} "
                f"· {self.scored}/{total} conversations scored",
                conversation.id,
            )

    async def save(self, rows: list[dict]) -> None:
        """A conversation's rows are written in one go, so a stopped run never leaves half of one."""
        lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        async with self.write_lock:
            with self.results.open("a", encoding="utf-8") as file:
                file.write(lines)

    def saved_rows(self) -> list[dict]:
        if not self.results.exists():
            return []
        return [
            json.loads(line)
            for line in self.results.read_text("utf-8").splitlines()
            if line.strip()
        ]

    async def close(self) -> None:
        for api in self.apis:
            await api.close()
        await llm.close()


async def run_phases(run: Run, to_run: list[dict]) -> None:
    log(f"phase 1/3: sending {len(to_run)} conversation(s)")
    async with asyncio.TaskGroup() as group:
        sending = [group.create_task(run.send(row)) for row in to_run]
    sent = [task.result() for task in sending]

    # No time limit on the whole wait: a large split takes hours. A turn Fibonna cannot compile, or
    # a compile queue that stops moving, stops the run.
    log(f"phase 2/3: waiting until Fibonna has compiled {len(sent)} conversation(s)")
    started = time.monotonic()
    progress = asyncio.create_task(run.log_processing_progress(len(sent), started))
    try:
        async with asyncio.TaskGroup() as group:
            for conversation in sent:
                group.create_task(run.wait(conversation, started))
    finally:
        progress.cancel()

    log(f"phase 3/3: scoring {len(sent)} conversation(s)")
    async with asyncio.TaskGroup() as group:
        for conversation in sent:
            group.create_task(run.score(conversation, len(sent)))


async def main_async(args: argparse.Namespace) -> None:
    missing = [
        name for name in REQUIRED_ENVIRONMENT if not os.environ.get(name, "").strip()
    ]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} before running (see README.md).")
    settings = Settings.from_environment()

    rows, revision = load_rows(args.split)
    if args.conversations is not None:
        rows = rows[: args.conversations]
    selected = [str(row.get("conversation_id", "")) for row in rows]
    if len(selected) > 1 and not settings.one_project_per_conversation:
        raise SystemExit(
            f"{len(selected)} conversations would share the project {settings.project!r}, and a recall "
            "would find the other conversations' turns. Put {conversation} in FIBONNA_BEAM_PROJECT "
            "to give each one a project, or run one conversation (see README.md)."
        )

    run = Run(args, settings)
    run.output.mkdir(parents=True, exist_ok=True)
    already_scored = {row["conversationId"] for row in run.saved_rows()} & set(selected)
    to_run = [
        row for row in rows if str(row.get("conversation_id", "")) not in already_scored
    ]
    log(
        f"BEAM {args.split} (dataset revision {revision[:12]}) on Fibonna: {len(selected)} conversation(s), "
        f"{len(already_scored)} already scored, {len(to_run)} to run · results in {run.results}"
    )

    # In every phase, a failure cancels the rest: a run finishes or stops, it never scores around
    # a gap. Rerunning resumes after the conversations already scored; turns already stored are
    # found again, not stored twice.
    try:
        if to_run:
            await run_phases(run, to_run)
    finally:
        await run.close()

    scored = [row for row in run.saved_rows() if row["conversationId"] in selected]
    summary = {
        "dataset": "BEAM",
        "split": args.split,
        "datasetRevision": revision,
        "conversations": len(selected),
        **scoring.summarize(scored),
        "answerModel": llm.MODEL,
        "judgeModel": llm.JUDGE_MODEL,
        "reasoningEffort": llm.REASONING_EFFORT,
        "recall": {"limit": RECALL_LIMIT, "maxTokens": RECALL_MAX_TOKENS},
        "prompts": {
            "source": exabase_prompts.SOURCE_URL,
            "sha256": exabase_prompts.SOURCE_SHA256,
        },
        "memory": "fibonna",
        "projectPerConversation": settings.one_project_per_conversation,
        "reusedProjects": run.reusing,
        "turnsSent": run.turns_sent,
        "compiledPages": run.compiled_pages,
        "harnessVersion": HARNESS_VERSION,
        "finishedAt": utc_iso(datetime.now(timezone.utc)),
    }
    (run.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", "utf-8"
    )
    print_summary(summary)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEAM against Fibonna's API.")
    parser.add_argument("--split", choices=SPLITS, default="100k")
    parser.add_argument(
        "--conversations", type=positive_number, help="only the first N conversations"
    )
    parser.add_argument(
        "--output", help="results directory (default: results/fibonna-<split>)"
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="ask again about conversations an earlier run sent and waited for: send nothing, wait for nothing",
    )
    parser.add_argument(
        "--compile-stall-minutes",
        type=positive_number,
        default=STALL_TIMEOUT_SECONDS // 60,
        help="stop when compiling has not moved for this long (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    try:
        asyncio.run(main_async(parse_arguments()))
    except* (ProjectMissing, CompileFailed) as stopped:
        raise SystemExit("; ".join(str(error) for error in stopped.exceptions))


if __name__ == "__main__":
    main()
