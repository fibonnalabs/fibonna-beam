"""The Fibonna adapter and runner, end to end and offline, against a local stand-in for the API.

    python -m unittest discover -s tests -t .      (from beam/)

The answer and judge model is replaced by two functions; nothing reaches the network.
"""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fibonna_api
import llm
import run_fibonna
from dataset import build_conversation, to_ingest_item
from fibonna_api import (
    CompileFailed,
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

        # Every turn was checked until chunked, and the queue was asked about until it was empty.
        self.assertEqual(pages, self.fibonna.pages)
        self.assertGreater(
            len(self.fibonna.calls("GET", r"/api/v1/knowledge/documents/\d+")),
            len(document_ids),
        )
        asked = [
            r["query"]["status"]
            for r in self.fibonna.calls("GET", "/api/v1/brain/compile/jobs")
        ]
        self.assertEqual(asked.count("pending"), 3 + fibonna_api.EMPTY_POLLS_TO_SETTLE)
        self.assertEqual(set(asked), {"pending", "running", "failed"})

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

    async def test_a_failed_compile_job_stops_the_wait(self) -> None:
        api = await self.api()
        document_ids = await self.ingest_conversation(api)
        self.fibonna.failed_compile_jobs = 1
        with self.assertRaises(CompileFailed):
            await api.wait_until_compiled(document_ids)

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

    def arguments(self, conversations: int | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            split="100k", conversations=conversations, output=str(self.output)
        )

    async def test_a_run_scores_every_question_in_its_own_project(self) -> None:
        await run_fibonna.main_async(self.arguments())

        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["questions"], 4)
        self.assertEqual(summary["microAverage"], 1.0)
        self.assertEqual(summary["failedQuestions"], 0)
        self.assertEqual(summary["turnsSent"], 8)
        self.assertEqual(summary["compiledPages"], {"7": 2, "8": 2})
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
