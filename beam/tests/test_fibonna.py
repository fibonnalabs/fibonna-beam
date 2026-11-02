"""The Fibonna adapter and runner, end to end and offline, against a local stand-in for the API.

    python -m unittest discover -s tests -t .      (from beam/)

The answer and judge model is replaced by two functions; nothing reaches the network.
"""

import argparse
import asyncio
import builtins
import json
import os
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import fibonna_api
import llm
import run_fibonna
from dataset import build_conversation, to_ingest_item
from fibonna_api import (
    CompileFailed,
    CompileTimedOut,
    FibonnaApi,
    FibonnaApiError,
    ProjectMissing,
    Settings,
)
from tests.fake_fibonna import TOKEN, FakeFibonna


def beam_row(conversation_id: str, first: str, second: str) -> dict:
    """A BEAM conversation in the dataset's own shape: two sessions, a changed fact, two questions."""
    return {
        "conversation_id": conversation_id,
        "conversation_seed": {"title": "Favourite colours"},
        "user_profile": {"user_info": "• Name: Craig Baker\n• Age: 34"},
        "chat": [
            [
                {
                    "role": "user",
                    "content": f"My favourite colour is {first}.",
                    "id": 1,
                    "time_anchor": "March-15-2024",
                },
                {
                    "role": "assistant",
                    "content": f"{first.title()} is a fine colour.",
                    "id": 2,
                },
            ],
            [
                {
                    "role": "user",
                    "content": f"Actually my favourite colour is {second}.",
                    "id": 3,
                    "time_anchor": "March-20-2024",
                },
                {"role": "assistant", "content": f"Noted, {second} it is.", "id": 4},
            ],
        ],
        "probing_questions": {
            "information_extraction": [
                {
                    "question": "What is my favourite colour?",
                    "ideal_response": second,
                    "rubric": [f"LLM response should contain: {second}"],
                }
            ],
            "knowledge_update": [
                {
                    "question": "What is my favourite colour now?",
                    "ideal_response": second,
                    "rubric": [f"LLM response should contain: {second}"],
                    "source_chat_ids": {"updated_info": [3]},
                }
            ],
        },
    }


class FibonnaTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.fibonna = FakeFibonna({"beam-7": 11, "beam-8": 12})
        self.fibonna.start()
        self.addCleanup(self.fibonna.stop)
        # Nothing to wait for in a stand-in: poll and retry at once.
        for name, value in (
            ("POLL_SECONDS", 0.001),
            ("RETRY_DELAYS_SECONDS", (0.001, 0.001, 0.001)),
        ):
            patcher = mock.patch.object(fibonna_api, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def api(self, project: str = "beam-7") -> FibonnaApi:
        api = FibonnaApi(self.fibonna.url, TOKEN, project)
        self.addAsyncCleanup(api.close)
        await api.open_project()
        return api

    async def until(self, condition: Callable[[], bool]) -> None:
        async with asyncio.timeout(10):
            while not condition():
                await asyncio.sleep(0.001)

    def compile_job_requests(self) -> int:
        return len(self.fibonna.calls("GET", "/api/v1/brain/compile/jobs"))


class AdapterTest(FibonnaTestCase):
    async def ingest_conversation(self, api: FibonnaApi) -> list[int]:
        conversation = build_conversation(beam_row("7", "red", "blue"))
        return [
            await api.ingest_turn(to_ingest_item(conversation, message))
            for message in conversation.messages
        ]

    async def test_ingest_wait_and_recall(self) -> None:
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        pages = await api.wait_until_compiled(document_ids)

        # The wait returned once each of the four turns had a compile job, and each had completed.
        self.assertEqual(pages, self.fibonna.pages + len(document_ids))
        self.assertEqual(len(self.fibonna.jobs), len(document_ids))
        self.assertEqual(self.fibonna.compiled_jobs(), len(document_ids))
        asked = {
            r["query"]["status"]
            for r in self.fibonna.calls("GET", "/api/v1/brain/compile/jobs")
        }
        self.assertEqual(asked, set(fibonna_api.COMPILE_JOB_STATUSES))
        for request in self.fibonna.calls("GET", "/api/v1/brain/compile/jobs"):
            self.assertEqual(request["query"]["project_id"], "11")

        # The end of the conversation: all four turns, as evidence text.
        recalled = await api.recall(
            "favourite colour noted",
            "2024-03-20T12:00:00.000003Z",
            identity="craig-baker-7",
        )
        texts = [result["text"] for result in recalled["results"]]
        self.assertEqual(len(texts), 4)
        self.assertTrue(
            any("Actually my favourite colour is blue." in text for text in texts)
        )
        self.assertEqual(recalled["asOf"], "2024-03-20T12:00:00.000003Z")
        # Each piece of evidence carries when it happened.
        self.assertTrue(
            all(r["occurredAt"].startswith("2024-03-") for r in recalled["results"])
        )

    async def test_recall_as_of_a_turn_includes_that_turn_and_nothing_after_it(
        self,
    ) -> None:
        api = await self.api()
        await self.ingest_conversation(api)
        # The third turn is at ...000002Z; the fourth, one microsecond later, must stay out.
        recalled = await api.recall(
            "favourite colour blue noted", "2024-03-20T12:00:00.000002Z"
        )
        texts = " ".join(result["text"] for result in recalled["results"])
        self.assertIn("Actually my favourite colour is blue.", texts)
        self.assertNotIn("Noted", texts)

    async def test_what_is_sent(self) -> None:
        api = await self.api()
        await self.ingest_conversation(api)
        await api.recall("favourite colour", "2024-03-20T12:00:00.000003Z")

        stored = self.fibonna.calls("POST", "/api/v1/ingest/text")[0]
        self.assertEqual(stored["body"]["project_id"], 11)
        self.assertEqual(stored["body"]["kb_type"], "project")
        self.assertEqual(stored["body"]["occurred_at"], "2024-03-15T12:00:00+00:00")
        self.assertEqual(stored["body"]["occurred_precision"], "exact")
        self.assertEqual(stored["body"]["title"], "Craig Baker (user) 7_s000_t0000")
        for request in self.fibonna.requests:
            self.assertEqual(request["headers"]["Authorization"], f"Bearer {TOKEN}")
            self.assertTrue(
                request["headers"]["User-Agent"].startswith(
                    "Mozilla/5.0 (fibonna-beam/"
                )
            )

        search = self.fibonna.calls("GET", r"/api/v1/knowledge/projects/11/search")[0][
            "query"
        ]
        self.assertEqual(search["top_k"], str(fibonna_api.RECALL_LIMIT))
        self.assertEqual(search["max_tokens"], str(fibonna_api.RECALL_MAX_TOKENS))
        self.assertEqual(search["include_personal"], "false")
        self.assertNotIn(
            "answer", search
        )  # the answer model reads the evidence, not Fibonna's own answer

    async def test_two_turns_with_the_same_words_stay_two_memories(self) -> None:
        api = await self.api()
        conversation = build_conversation(beam_row("7", "red", "blue"))
        first = to_ingest_item(conversation, conversation.messages[0])
        second = {
            **first,
            "id": "7_s000_t0099",
            "timestamp": "2024-03-15T12:00:00.000099+00:00",
        }
        self.assertNotEqual(await api.ingest_turn(first), await api.ingest_turn(second))

    async def test_a_turn_sent_again_is_found_not_stored_twice(self) -> None:
        api = await self.api()
        conversation = build_conversation(beam_row("7", "red", "blue"))
        item = to_ingest_item(conversation, conversation.messages[0])
        self.assertEqual(await api.ingest_turn(item), await api.ingest_turn(item))
        self.assertEqual(len(self.fibonna.documents), 1)

    async def test_a_server_error_is_retried_and_a_refusal_is_not(self) -> None:
        api = await self.api()
        self.fibonna.failures["POST /api/v1/ingest/text"] = [503, 429]
        await self.ingest_conversation(api)
        self.assertEqual(len(self.fibonna.documents), 4)
        self.assertEqual(len(self.fibonna.calls("POST", "/api/v1/ingest/text")), 6)

        self.fibonna.failures["GET /api/v1/brain/index"] = [403]
        with self.assertRaises(FibonnaApiError) as refused:
            await api.brain_index()
        self.assertEqual(refused.exception.status, 403)
        self.assertEqual(len(self.fibonna.calls("GET", "/api/v1/brain/index")), 1)

    async def test_a_wrong_token_is_refused(self) -> None:
        api = FibonnaApi(self.fibonna.url, "wrong", "beam-7")
        self.addAsyncCleanup(api.close)
        with self.assertRaises(FibonnaApiError) as refused:
            await api.open_project()
        self.assertEqual(refused.exception.status, 401)

    async def test_a_project_that_does_not_exist_is_not_created(self) -> None:
        api = FibonnaApi(self.fibonna.url, TOKEN, "beam-99")
        self.addAsyncCleanup(api.close)
        with self.assertRaises(ProjectMissing):
            await api.open_project()
        self.assertEqual({r["method"] for r in self.fibonna.requests}, {"GET"})

    async def test_a_job_that_failed_or_was_skipped_stops_the_wait(self) -> None:
        for final_status in ("failed", "skipped"):
            with self.subTest(final_status):
                self.fibonna.final_status = final_status
                api = await self.api()
                document_ids = await self.ingest_conversation(api)
                with self.assertRaises(CompileFailed) as stopped:
                    await api.wait_until_compiled(document_ids)
                self.assertNotIsInstance(stopped.exception, CompileTimedOut)
                self.assertIn(final_status, str(stopped.exception))
                self.assertIn("document", str(stopped.exception))
                if final_status == "failed":
                    self.assertIn("boom", str(stopped.exception))
                self.fibonna.documents.clear()
                self.fibonna.jobs.clear()

    async def test_a_turn_dated_by_its_arrival_is_refused(self) -> None:
        api = await self.api()
        self.fibonna.keep_event_time = False
        conversation = build_conversation(beam_row("7", "red", "blue"))
        with self.assertRaises(FibonnaApiError):
            await api.ingest_turn(
                to_ingest_item(conversation, conversation.messages[0])
            )

    async def test_a_search_that_ignores_the_moment_is_refused(self) -> None:
        api = await self.api()
        await self.ingest_conversation(api)
        self.fibonna.echo_window = False
        with self.assertRaises(FibonnaApiError):
            await api.recall("favourite colour", "2024-03-20T12:00:00.000003Z")


class CompileBarrierTest(FibonnaTestCase):
    """A turn is compiled when its own compile job has completed, and not before it exists."""

    async def ingest_conversation(self, api: FibonnaApi) -> list[int]:
        conversation = build_conversation(beam_row("7", "red", "blue"))
        return [
            await api.ingest_turn(to_ingest_item(conversation, message))
            for message in conversation.messages
        ]

    async def test_the_wait_holds_while_no_compile_job_is_queued_yet(self) -> None:
        # The turns are stored and chunked, and their compile jobs are queued a while later.
        self.fibonna.hold_compile_enqueue = True
        api = await self.api()
        document_ids = await self.ingest_conversation(api)

        waiting = asyncio.create_task(api.wait_until_compiled(document_ids))
        self.addCleanup(waiting.cancel)
        # The queue is asked about several times, and has no job for any turn: the wait stays.
        asked_before = self.compile_job_requests()
        await self.until(
            lambda: (
                self.compile_job_requests()
                >= asked_before + 3 * len(fibonna_api.COMPILE_JOB_STATUSES)
            )
        )
        self.assertFalse(waiting.done())
        self.assertEqual(self.fibonna.jobs, {})

        self.fibonna.release_compile_enqueue()
        pages = await asyncio.wait_for(waiting, 10)
        # It returned after the jobs were queued, and only when each had completed.
        self.assertEqual(len(self.fibonna.jobs), len(document_ids))
        self.assertEqual(self.fibonna.compiled_jobs(), len(document_ids))
        self.assertEqual(pages, self.fibonna.pages + len(document_ids))

    async def test_a_turn_still_without_a_job_is_waited_for_when_the_others_are_done(
        self,
    ) -> None:
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        # The first three have completed jobs; the fourth's job is not queued yet.
        self.fibonna.hold_compile_enqueue = True
        conversation = build_conversation(beam_row("7", "red", "blue"))
        late = await api.ingest_turn(
            {
                **to_ingest_item(conversation, conversation.messages[0]),
                "id": "7_late",
                "timestamp": "2024-03-21T12:00:00+00:00",
            }
        )

        waiting = asyncio.create_task(api.wait_until_compiled([*document_ids, late]))
        self.addCleanup(waiting.cancel)
        await self.until(lambda: self.fibonna.compiled_jobs() == len(document_ids))
        asked_before = self.compile_job_requests()
        await self.until(lambda: self.compile_job_requests() >= asked_before + 20)
        self.assertFalse(waiting.done())

        self.fibonna.release_compile_enqueue()
        await asyncio.wait_for(waiting, 10)
        self.assertEqual(self.fibonna.compiled_jobs(), len(document_ids) + 1)

    async def test_a_turn_that_never_gets_a_job_times_out_naming_it(self) -> None:
        self.fibonna.hold_compile_enqueue = True
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        with self.assertRaises(CompileTimedOut) as stopped:
            await api.wait_until_compiled(document_ids, stall_timeout=0.05)
        message = str(stopped.exception)
        self.assertIn("beam-7", message)
        self.assertIn("4 with no compile job", message)
        self.assertIn(str(document_ids[0]), message)
        # It is a CompileFailed, which is what stops the runner with a message.
        self.assertIsInstance(stopped.exception, CompileFailed)

    async def test_a_job_that_never_finishes_times_out(self) -> None:
        self.fibonna.drains = False
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        with self.assertRaises(CompileTimedOut) as stopped:
            await api.wait_until_compiled(document_ids, stall_timeout=0.05)
        self.assertIn("4 pending", str(stopped.exception))

    async def test_a_wait_is_not_stopped_while_the_other_conversations_move(
        self,
    ) -> None:
        # The queue takes the newest jobs first, so the conversation sent first can stand still for
        # as long as the others take to compile. Its jobs here never move.
        self.fibonna.drains = False
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        clock = fibonna_api.StallClock()
        waiting = asyncio.create_task(
            api.wait_until_compiled(document_ids, stall_timeout=0.3, clock=clock)
        )
        self.addCleanup(waiting.cancel)
        for _ in range(12):  # twice the stall timeout
            await asyncio.sleep(0.05)
            clock.moved()
        self.assertFalse(waiting.done())

        # Once nothing moves anywhere, it is stopped.
        with self.assertRaises(CompileTimedOut) as stopped:
            await asyncio.wait_for(waiting, 5)
        self.assertIn("4 pending", str(stopped.exception))

    async def test_a_wait_that_is_moving_is_not_timed_out(self) -> None:
        # Four turns whose jobs each take two listings: slower than the stall timeout in all, but
        # never still for that long.
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        with mock.patch.object(fibonna_api, "POLL_SECONDS", 0.02):
            await api.wait_until_compiled(document_ids, stall_timeout=0.1)
        self.assertEqual(self.fibonna.compiled_jobs(), len(document_ids))

    async def test_a_job_another_turn_owns_does_not_complete_this_turn(self) -> None:
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        # Only the first turn's job is queued; the others' are held back.
        stray = next(iter(self.fibonna.jobs.values()))
        del self.fibonna.jobs[4], self.fibonna.jobs[3], self.fibonna.jobs[2]
        self.assertEqual(stray["source_id"], str(document_ids[0]))
        with self.assertRaises(CompileTimedOut) as stopped:
            await api.wait_until_compiled(document_ids, stall_timeout=0.05)
        self.assertIn("3 with no compile job", str(stopped.exception))

    async def test_jobs_beyond_the_newest_page_are_still_counted(self) -> None:
        # The jobs route lists the newest `limit` of a status only. Jobs that finished before the
        # wait began fall out of the newest two, but were seen while the turns were sent. No source
        # page helps here: the jobs alone show the turns compiled.
        self.fibonna.instant_drain = True
        self.fibonna.source_pages = False
        api = await self.api()
        with (
            mock.patch.object(fibonna_api, "COMPILE_JOBS_LIMIT", 2),
            mock.patch.object(fibonna_api, "OBSERVE_EVERY_TURNS", 1),
        ):
            document_ids = await self.ingest_conversation(api)
            pages = await api.wait_until_compiled(document_ids, stall_timeout=1)
        self.assertEqual(self.fibonna.compiled_jobs(), len(document_ids))
        self.assertEqual(pages, self.fibonna.pages)

    async def test_a_project_compiled_in_full_can_be_waited_for_again(self) -> None:
        # A run that stopped is started again: its turns are sent again, each found and not stored
        # twice, into a project whose jobs have mostly left the newest of their status.
        self.fibonna.instant_drain = True
        await self.ingest_conversation(await self.api())
        again = await self.api()
        with mock.patch.object(fibonna_api, "COMPILE_JOBS_LIMIT", 2):
            document_ids = await self.ingest_conversation(again)
            pages = await again.wait_until_compiled(document_ids, stall_timeout=1)
        self.assertEqual(len(self.fibonna.documents), 4)
        self.assertEqual(len(self.fibonna.jobs), 4)
        self.assertEqual(pages, self.fibonna.pages + 4)

    async def test_a_timeout_says_when_older_jobs_may_be_out_of_sight(self) -> None:
        # The turns were stored and compiled before the wait began, with no look at the queue
        # between, and compiling left no source page for them: nothing shows they were compiled.
        self.fibonna.instant_drain = True
        self.fibonna.source_pages = False
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        with (
            mock.patch.object(fibonna_api, "COMPILE_JOBS_LIMIT", 2),
            self.assertRaises(CompileTimedOut) as stopped,
        ):
            await api.wait_until_compiled(document_ids, stall_timeout=0.05)
        self.assertIn("2 with no compile job", str(stopped.exception))
        self.assertIn("newest 2 jobs", str(stopped.exception))
        self.assertIn("source page", str(stopped.exception))

    async def test_a_source_page_of_another_turn_does_not_complete_this_turn(
        self,
    ) -> None:
        self.fibonna.hold_compile_enqueue = True
        # Turn 12's page, and a page whose name merely starts with turn 1's id.
        self.fibonna.extra_slugs = [
            "src-document-12-quick-note-turn-md",
            "src-document-10",
        ]
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        self.assertEqual(document_ids, [1, 2, 3, 4])
        with self.assertRaises(CompileTimedOut) as stopped:
            await api.wait_until_compiled(document_ids, stall_timeout=0.05)
        self.assertIn("4 with no compile job", str(stopped.exception))

    async def test_a_queued_job_that_completes_out_of_sight_is_settled(self) -> None:
        # A job that is retried finishes after newer ones, so the newest completed jobs no longer
        # reach it: here the first turn's job, seen pending while the turns were sent.
        self.fibonna.drains = False
        self.fibonna.source_pages = False
        api = await self.api()
        with (
            mock.patch.object(fibonna_api, "COMPILE_JOBS_LIMIT", 3),
            mock.patch.object(fibonna_api, "OBSERVE_EVERY_TURNS", 1),
        ):
            document_ids = await self.ingest_conversation(api)
            for job in self.fibonna.jobs.values():
                job["status"] = "completed"
            pages = await api.wait_until_compiled(document_ids, stall_timeout=1)
        self.assertEqual(self.fibonna.compiled_jobs(), len(document_ids))
        self.assertEqual(pages, self.fibonna.pages)

    async def test_a_queued_job_a_full_listing_may_hide_is_not_taken_for_finished(
        self,
    ) -> None:
        # Two newer pending jobs of another kind fill the listing, and the turn's job is not in it.
        self.fibonna.drains = False
        api = await self.api()
        conversation = build_conversation(beam_row("7", "red", "blue"))
        with (
            mock.patch.object(fibonna_api, "COMPILE_JOBS_LIMIT", 2),
            mock.patch.object(fibonna_api, "OBSERVE_EVERY_TURNS", 1),
        ):
            document_id = await api.ingest_turn(
                to_ingest_item(conversation, conversation.messages[0])
            )
            for job_id in (98, 99):
                self.fibonna.jobs[job_id] = {
                    "id": job_id,
                    "project_id": 11,
                    "source_type": "project_pulse",
                    "source_id": "11",
                    "status": "pending",
                    "error": None,
                }
            with self.assertRaises(CompileTimedOut) as stopped:
                await api.wait_until_compiled([document_id], stall_timeout=0.1)
        self.assertIn("1 pending", str(stopped.exception))

    async def test_a_job_missing_from_one_round_is_not_taken_for_finished(
        self,
    ) -> None:
        # A retry takes a job from running back to pending between two of the asks of a round, and
        # no listing of that round shows it. The next round does.
        self.fibonna.drains = False
        self.fibonna.source_pages = False
        api = await self.api()
        conversation = build_conversation(beam_row("7", "red", "blue"))
        with mock.patch.object(fibonna_api, "OBSERVE_EVERY_TURNS", 1):
            document_id = await api.ingest_turn(
                to_ingest_item(conversation, conversation.messages[0])
            )
        self.fibonna.drains = True
        self.fibonna.hidden[1] = len(fibonna_api.COMPILE_JOB_STATUSES)
        await api.wait_until_compiled([document_id], stall_timeout=1)
        self.assertEqual(self.fibonna.jobs[1]["status"], "completed")

    async def test_the_jobs_of_other_sources_are_left_alone(self) -> None:
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        # A project pulse job shares the queue; it has no document, and its failure is not a turn's.
        self.fibonna.jobs[99] = {
            "id": 99,
            "project_id": 11,
            "source_type": "project_pulse",
            "source_id": "11",
            "status": "failed",
            "error": "pulse",
        }
        pages = await api.wait_until_compiled(document_ids)
        self.assertEqual(pages, self.fibonna.pages + len(document_ids))
        self.assertEqual(self.fibonna.jobs[99]["status"], "failed")


class SettingsTest(unittest.TestCase):
    def test_everything_comes_from_the_environment(self) -> None:
        names = {
            "FIBONNA_API_URL": " http://x ",
            "FIBONNA_API_TOKEN": "t",
            "FIBONNA_BEAM_PROJECT": "beam-{conversation}",
        }
        with mock.patch.dict(os.environ, names):
            settings = Settings.from_environment()
        self.assertEqual((settings.base_url, settings.token), ("http://x", "t"))
        self.assertTrue(settings.one_project_per_conversation)
        self.assertEqual(settings.project_for("12"), "beam-12")

    def test_a_missing_variable_stops_the_run(self) -> None:
        names = {"FIBONNA_API_URL": "http://x", "FIBONNA_API_TOKEN": "t"}
        with (
            mock.patch.dict(os.environ, names, clear=True),
            self.assertRaises(SystemExit) as stopped,
        ):
            Settings.from_environment()
        self.assertIn("FIBONNA_BEAM_PROJECT", str(stopped.exception))


class RunnerTest(FibonnaTestCase):
    """run_fibonna end to end: ingest, wait, recall, answer, judge, summary."""

    def setUp(self) -> None:
        super().setUp()
        self.prompts: list[str] = []

        async def complete_text(prompt: str) -> str:
            self.prompts.append(prompt)
            return "Your favourite colour is blue."

        async def complete_json(prompt: str, schema: dict) -> dict:
            return {"score": 1.0, "reason": "it says blue"}

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "out"
        rows = [beam_row("7", "red", "blue"), beam_row("8", "green", "yellow")]
        environment = {
            "FIBONNA_API_URL": self.fibonna.url,
            "FIBONNA_API_TOKEN": TOKEN,
            "FIBONNA_BEAM_PROJECT": "beam-{conversation}",
            "OPENROUTER_API_KEY": "unused",
        }
        for patcher in (
            mock.patch.dict(os.environ, environment),
            mock.patch.object(llm, "complete_text", complete_text),
            mock.patch.object(llm, "complete_json", complete_json),
            mock.patch.object(
                run_fibonna, "load_rows", lambda split: (rows, "abc123abc123abc123")
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def arguments(
        self,
        conversations: int | None = None,
        *,
        reuse: bool = False,
        output: Path | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            split="100k",
            conversations=conversations,
            output=str(output or self.output),
            compile_stall_minutes=30,
            reuse=reuse,
        )

    async def test_a_run_scores_every_question_in_its_own_project(self) -> None:
        await run_fibonna.main_async(self.arguments())

        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["questions"], 4)
        self.assertEqual(summary["microAverage"], 1.0)
        self.assertEqual(summary["failedQuestions"], 0)
        self.assertEqual(summary["turnsSent"], 8)
        # Two pages of another kind and a source page for each of the conversation's four turns.
        self.assertEqual(summary["compiledPages"], {"7": 6, "8": 6})
        self.assertEqual(summary["memory"], "fibonna")
        self.assertTrue(summary["projectPerConversation"])

        rows = [
            json.loads(line)
            for line in (self.output / "results.jsonl").read_text().splitlines()
        ]
        evidence = {
            row["questionId"]: " ".join(
                result["text"] for result in row["recall"]["results"]
            )
            for row in rows
        }
        # Each conversation recalls its own turns and none of the other's.
        self.assertIn("blue", evidence["7_information_extraction_0"])
        self.assertNotIn("green", evidence["7_information_extraction_0"])
        self.assertIn("yellow", evidence["8_information_extraction_0"])
        self.assertNotIn("blue", evidence["8_information_extraction_0"])
        # A knowledge_update question is asked as of the update: that turn, and not the reply after it.
        self.assertIn(
            "Actually my favourite colour is blue.", evidence["7_knowledge_update_0"]
        )
        self.assertNotIn("Noted", evidence["7_knowledge_update_0"])
        self.assertIn("Noted, blue it is.", evidence["7_information_extraction_0"])
        # The answer model was handed that evidence inside ExaBase's prompt.
        self.assertTrue(
            any(
                "Actually my favourite colour is blue." in prompt
                for prompt in self.prompts
            )
        )

    async def test_nothing_is_asked_before_every_turn_has_compiled(self) -> None:
        # The compile jobs are queued only after the turns are stored: the run waits for them.
        self.fibonna.hold_compile_enqueue = True
        running = asyncio.create_task(run_fibonna.main_async(self.arguments()))
        self.addCleanup(running.cancel)
        await self.until(lambda: len(self.fibonna.documents) == 8)
        asked_before = self.compile_job_requests()
        await self.until(lambda: self.compile_job_requests() >= asked_before + 20)
        self.assertFalse(running.done())
        self.assertEqual(
            self.fibonna.calls("GET", r"/api/v1/knowledge/projects/\d+/search"), []
        )

        self.fibonna.release_compile_enqueue()
        await asyncio.wait_for(running, 30)
        self.assertEqual(self.fibonna.compiled_at_first_search, 8)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["failedQuestions"], 0)

    async def test_every_conversation_waits_on_one_stall_clock(self) -> None:
        clocks = []
        wait = FibonnaApi.wait_until_compiled

        async def spy(api, document_ids, conversation, stall_timeout, clock):
            clocks.append(clock)
            return await wait(api, document_ids, conversation, stall_timeout, clock)

        with mock.patch.object(FibonnaApi, "wait_until_compiled", spy):
            await run_fibonna.main_async(self.arguments())
        self.assertEqual(len(clocks), 2)
        self.assertIsInstance(clocks[0], fibonna_api.StallClock)
        self.assertIs(clocks[0], clocks[1])

    async def test_a_run_whose_compile_stops_moving_is_stopped_with_a_message(
        self,
    ) -> None:
        self.fibonna.hold_compile_enqueue = True
        with mock.patch.object(fibonna_api, "STALL_TIMEOUT_SECONDS", 0.05):
            arguments = self.arguments(conversations=1)
            arguments.compile_stall_minutes = 0.05 / 60
            with self.assertRaises(builtins.ExceptionGroup) as stopped:
                await run_fibonna.main_async(arguments)
        self.assertTrue(stopped.exception.subgroup(CompileTimedOut))
        self.assertEqual(
            self.fibonna.calls("GET", r"/api/v1/knowledge/projects/\d+/search"), []
        )

    async def test_a_second_run_asks_again_without_sending_or_waiting(self) -> None:
        await run_fibonna.main_async(self.arguments())
        stored = len(self.fibonna.calls("POST", "/api/v1/ingest/text"))
        jobs_asked = self.compile_job_requests()

        again = self.output.parent / "again"
        await run_fibonna.main_async(self.arguments(reuse=True, output=again))

        self.assertEqual(len(self.fibonna.calls("POST", "/api/v1/ingest/text")), stored)
        self.assertEqual(len(self.fibonna.documents), 8)
        summary = json.loads((again / "summary.json").read_text())
        self.assertEqual(
            (summary["questions"], summary["failedQuestions"], summary["turnsSent"]),
            (4, 0, 0),
        )
        self.assertTrue(summary["reusedProjects"])
        self.assertEqual(summary["compiledPages"], {"7": 6, "8": 6})
        # The queue was looked at once for each of the two projects: five statuses each.
        self.assertEqual(
            self.compile_job_requests() - jobs_asked,
            2 * len(fibonna_api.COMPILE_JOB_STATUSES),
        )

    async def test_a_second_run_refuses_a_project_that_is_still_compiling(self) -> None:
        await run_fibonna.main_async(self.arguments())
        self.fibonna.jobs[1]["status"] = "running"
        self.fibonna.drains = False  # it stays running
        again = self.output.parent / "again"
        with self.assertRaises(builtins.ExceptionGroup) as stopped:
            await run_fibonna.main_async(self.arguments(reuse=True, output=again))
        self.assertIn("not compiled", str(stopped.exception.exceptions[0]))
        self.assertFalse((again / "results.jsonl").exists())

    async def test_a_second_run_refuses_a_project_with_no_wiki_pages(self) -> None:
        await run_fibonna.main_async(self.arguments(conversations=1))
        self.fibonna.pages = 0
        self.fibonna.source_pages = False
        again = self.output.parent / "again"
        with self.assertRaises(builtins.ExceptionGroup) as stopped:
            await run_fibonna.main_async(
                self.arguments(conversations=1, reuse=True, output=again)
            )
        self.assertIn("no wiki pages", str(stopped.exception.exceptions[0]))

    async def test_a_run_resumes_after_the_conversations_already_scored(self) -> None:
        await run_fibonna.main_async(self.arguments(conversations=1))
        stored = len(self.fibonna.documents)
        await run_fibonna.main_async(self.arguments())

        rows = [
            json.loads(line)
            for line in (self.output / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            sorted(row["conversationId"] for row in rows), ["7", "7", "8", "8"]
        )
        self.assertEqual(len(self.fibonna.documents), stored + 4)

    async def test_conversations_do_not_share_a_project_by_accident(self) -> None:
        with mock.patch.dict(os.environ, {"FIBONNA_BEAM_PROJECT": "beam-7"}):
            with self.assertRaises(SystemExit) as stopped:
                await run_fibonna.main_async(self.arguments())
            self.assertIn("{conversation}", str(stopped.exception))
            self.assertEqual(self.fibonna.requests, [])
            # One conversation alone owns the project.
            await run_fibonna.main_async(self.arguments(conversations=1))
        self.assertEqual(len(self.fibonna.documents), 4)

    async def test_a_failed_recall_scores_zero_and_the_run_goes_on(self) -> None:
        self.fibonna.echo_window = False
        await run_fibonna.main_async(self.arguments(conversations=1))
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(
            (summary["questions"], summary["failedQuestions"], summary["microAverage"]),
            (2, 2, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
