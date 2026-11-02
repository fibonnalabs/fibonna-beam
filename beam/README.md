# BEAM on past.dev

This harness runs the [BEAM](https://arxiv.org/abs/2510.27246) long-conversation memory benchmark
against past.dev's public API. It loads the dataset, ingests each conversation, waits for
past.dev to process it, then asks the benchmark's questions and scores the answers.

The answer and judge prompts are ExaBase's published BEAM prompts, unchanged, so scores compare
with other published BEAM results. Everything else is ordinary use of the public API.

## Results

The saved baseline covers all 100 conversations and 2,000 questions. The combined score,
weighted by question count, is **90.02%**.

| Split | Conversations | Questions | Score | Files |
|---|---:|---:|---:|---|
| 100K | 20 | 400 | **92.08%** | [Summary](results/100k/summary.json) · [Results](results/100k/results.jsonl) |
| 500K | 35 | 700 | **89.63%** | [Summary](results/500k/summary.json) · [Results](results/500k/results.jsonl) |
| 1M | 35 | 700 | **90.65%** | [Summary](results/1m/summary.json) · [Results](results/1m/results.jsonl) |
| 10M | 10 | 200 | **85.03%** | [Summary](results/10m/summary.json) · [Results](results/10m/results.jsonl) |

| Category | 100K | 500K | 1M | 10M |
|---|---:|---:|---:|---:|
| Abstention | 98.75% | 92.14% | 97.14% | 100.00% |
| Contradiction resolution | 78.13% | 78.57% | 75.36% | 73.13% |
| Event ordering | 100.00% | 100.00% | 99.81% | 100.00% |
| Information extraction | 95.00% | 86.32% | 83.98% | 68.75% |
| Instruction following | 96.25% | 92.02% | 96.43% | 95.00% |
| Knowledge update | 89.38% | 86.07% | 85.71% | 92.50% |
| Multi-session reasoning | 68.95% | 71.39% | 72.82% | 28.17% |
| Preference following | 100.00% | 99.05% | 98.87% | 100.00% |
| Summarization | 99.38% | 97.25% | 99.60% | 99.00% |
| Temporal reasoning | 95.00% | 93.45% | 96.79% | 93.75% |

Each split includes a summary and per-question results with answers, evidence, scores, and
judge verdicts. Use `verify.py` to check the scores against the saved verdicts.

## Run it

You need Python 3.11 or newer.

```bash
pip install -r requirements.txt
export PAST_API_URL=https://api.past.dev
export PAST_MANAGEMENT_KEY=past_mk_...        # Settings › Organization key
export OPENROUTER_API_KEY=...
python run.py --split 100k                    # 100k, 500k, 1m or 10m
python run.py --split 100k --conversations 5  # only the first five conversations
```

- **Projects.** Each conversation gets its own project: 20 at 100K, 35 at 500K and 1M, 10 at 10M.
  If your account's project limit is lower, use `--conversations N`. The harness never deletes
  projects.
- **Dataset.** Each split downloads from Hugging Face once and is cached in `~/.cache/beam-open`
  (set `BEAM_OPEN_CACHE` to move it). The 10M split is about 1 GiB.
- **New run results** go to `results/beam-<split>/`. `results.jsonl` has one line per question, with the
  recall response, the answer and the judge's verdicts. `summary.json` has the scores, the dataset
  revision and the models used.
- **Resuming.** Run the same command again. Conversations already scored are skipped, and one that
  was cut off starts over in a new project.

### Score again without reingesting

Each run tags its projects with its start time, shown in the first log line and in
`summary.json`. Pass the tag to recall and score the same projects again:

```bash
python run.py --split 100k --reuse 0923-164412 --output results/beam-100k-rescore
```

Nothing is ingested again. The harness waits until the projects are processed, asks every question,
and removes the temporary keys it created. To find what to wait on, it resends each conversation's
last batch, which past.dev recognizes by its idempotency key and neither ingests nor charges again.
Use the same harness code and dataset revision as the original run. Changes to ingestion
payloads change batch idempotency keys and require fresh projects. If replaying a batch returns
an idempotency conflict, the compatibility fallback assumes the project has finished processing;
it cannot verify the original push's status.

## Running against Fibonna

`run_fibonna.py` runs the same benchmark against a Fibonna stack. The dataset, ExaBase's prompts,
the scoring (`scoring.py`) and `verify.py` are the ones above, unchanged; only where the memory
lives differs. `fibonna_api.py` is the adapter, in the shape of `past_api.py`: it ingests a turn,
polls until compile has finished, and recalls through search, returning the evidence text.

**Run it only on a throwaway stack.** Never on the live stack or on a pilot's stack. The run writes
thousands of memories and cannot take them back, so give it projects made for it, and delete the
stack when you are done.

### Environment

| Variable | What it is |
|---|---|
| `FIBONNA_API_URL` | the stack's API, e.g. `https://api.example.test` |
| `FIBONNA_API_TOKEN` | a token that can read and write the benchmark projects |
| `FIBONNA_BEAM_PROJECT` | the benchmark project's slug; with `{conversation}` in it, one project per conversation (below) |
| `OPENROUTER_API_KEY` | the answer and judge model, as above |
| `BEAM_JUDGE_MODEL` | optional: an OpenRouter model id for the judge only (see "A second judge") |

Nothing is built in: no URL, token or project name is in the code. The projects must exist
already; the harness never creates one.

**One project per conversation.** past.dev's run gives each conversation its own project, so a
recall finds only that conversation's turns. Do the same: create one project per conversation,
named by the conversation's id (`beam-100k-1` to `beam-100k-20`), and set
`FIBONNA_BEAM_PROJECT='beam-100k-{conversation}'`. With a slug that has no `{conversation}`, the
harness refuses to run more than one conversation, because they would share one memory and each
question would be answered from the others' turns too.

The stack needs: the compile drain running (the gateway's schedule), and search accepting up to
100 results and an 8,000-token evidence budget, which is past.dev's recall budget (the setting
behind `search_evidence_budget_enabled`). It uses these product routes, and no others:
`GET /api/v1/projects/resolve`, `POST /api/v1/ingest/text`, `GET /api/v1/knowledge/documents/{id}`,
`GET /api/v1/brain/compile/jobs`, `GET /api/v1/brain/index` and
`GET /api/v1/knowledge/projects/{id}/search`.

### How a conversation is sent and asked

- **Turns.** Each turn is one project memory, in order, with BEAM's date for it as its event time
  (`occurred_at`, to the microsecond, as in the Past run). Its title is the author and the turn's
  id, which keeps two turns with the same words as two memories. Sending a turn again finds the
  memory already stored, so a run that stopped can be started again without storing anything twice.
- **Waiting.** The harness polls until every turn has been chunked, then until the project's
  compile queue has had no pending or running job for two polls in a row, and logs the number of
  wiki pages the compile produced. A compile job that failed stops the run.
- **Asking.** Search runs as of the moment the question is asked, the end of the conversation or
  for knowledge-update questions the update the dataset cites, and the turn at that moment is
  included. The answer model reads the evidence search returns, in ExaBase's answer prompt; the
  answer Fibonna's own search can write is not asked for.

### Dry run: one conversation

Conversation 1 is the smallest: 188 turns (about 131,000 tokens) and 20 questions.

```bash
pip install -r requirements.txt
export FIBONNA_API_URL=...             # the throwaway stack
export FIBONNA_API_TOKEN=...
export FIBONNA_BEAM_PROJECT=beam-dry   # a project you created for this
export OPENROUTER_API_KEY=...
python run_fibonna.py --split 100k --conversations 1 --output results/fibonna-dry
python verify.py results/fibonna-dry
```

Before the full run, check in the output that `failedQuestions` and `failedVerdicts` in
`summary.json` are 0, that `compiledPages` is not 0, and that the evidence in a few `recall`
records is from that conversation. Then read the compile cost per turn the stack reports and
replace the estimate below with it.

### The 100K run

20 conversations, 400 questions (40 in each of the ten categories), 5,732 turns and about
3.06 million tokens. Create the 20 projects, then:

```bash
export FIBONNA_BEAM_PROJECT='beam-100k-{conversation}'
python run_fibonna.py --split 100k
python verify.py results/fibonna-100k
```

Results go to `results/fibonna-100k/`, which git ignores. Run the command again to resume: scored
conversations are skipped. Run it with another `--output` to ask every question again, for example
with another judge; the turns are found in their projects, not stored again, so nothing compiles a
second time.

### Compile cost estimate

Every turn is a memory and each memory is compiled once. The compile makes up to seven model calls
and costs $0.14 to $0.16 on Sonnet 5, the internal design estimate (not measured).

| Run | Turns | Compiles at $0.14 to $0.16 |
|---|---:|---:|
| Dry run, conversation 1 | 188 | $26 to $30 |
| 100K split, 20 conversations | 5,732 | $800 to $920 |

This is the compile alone. The answers and the judge go through OpenRouter and are charged
separately: one answer per question and one judge call per rubric item. One turn, in conversation
4, is 348,469 characters (about 87,000 tokens), far larger than the rest (the next is 53,107); it
may compile for more than the average, and a stack with a request size limit may refuse it.

### A second judge

past.dev's run has one model answer and judge. To also report a judge from another family, set
`BEAM_JUDGE_MODEL` to the OpenRouter id of a model that is not OpenAI's. Only the judge's calls use
it: the answers still come from `openai/gpt-5.6-luna`, and `summary.json` records both models.
Report the two runs side by side, with the model ids. Try it in the dry run first: the judge is
asked for strict JSON, which not every provider enforces.

### Licences

A Fibonna run's results hold dataset questions, reference answers and recalled excerpts, so they
keep the dataset's CC BY-SA 4.0 terms, and ExaBase's prompts stay unchanged and credited; see
[NOTICE.md](NOTICE.md) before publishing any of it.

### Tests

```bash
python -m unittest discover -s tests -t .
```

They run the adapter and the runner against a stand-in for the Fibonna API on a local port, and a
stand-in for the model. Nothing reaches the network.

## Check a run

```bash
python verify.py results/100k        # verify the published 100K results
python verify.py --prompts           # the prompts match ExaBase's published file
```

The first check makes no Past or model API calls. It loads the public dataset (downloading it if
not already cached), confirms that every expected question has one record, checks each score
against the saved judge verdicts, and confirms that the summary is their mean. It verifies the
recorded scoring arithmetic; it does not rerun retrieval, answer generation, or judging. The second
downloads ExaBase's adapter, checks its SHA-256 (`584b672d…fc43`) and confirms our prompts match it.

## How answers are scored

- The answer model reads the question and past.dev's recall response, as returned, inside
  ExaBase's answer prompt.
- The judge is ExaBase's version of the BEAM paper's judge (`scoring.py`):
  - **Event ordering:** the answer's list is matched to the reference topics, and the two orders
    are compared with Kendall tau-b, scaled from 0 to 1.
  - **Every other category:** each rubric item scores 0, 0.5 or 1, and the question scores their
    average.
  - The headline is the mean over all questions. The summary also gives each category's mean.
- One model answers and judges: `openai/gpt-5.6-luna` through OpenRouter, with reasoning off. Each
  question is answered once per harness invocation.
- A call that still fails after retries scores 0, as in ExaBase's adapter, and `summary.json`
  counts these. Only an account error, such as a wrong key or no credits, stops the run.

## How conversations are sent

- **Identities.** The user's identity is their name plus the conversation id (`craig-baker-3`),
  with their profile as traits. The assistant is `assistant`. Each turn is sent as its author, and
  every question is asked as the user.
- **Turns.** Each turn carries a source ID, content, label, timestamp, and identity. Its metadata
  contains only role, author, and speaker. Turns go in pushes of up to 100
  (`POST /api/v1/ingest/batch`), in order. BEAM
  dates sessions by day, so each turn is placed at noon UTC on its day, one microsecond after the
  turn before it. Nothing sent names the benchmark except the project and key names.
- **Waiting.** The harness polls each conversation's last push (`GET /api/v1/ingest/{id}`) until it
  is `completed` and the project is `settled`. A push that fails stops the run. The 10M split takes
  hours.
- **Asking.** Recall runs at the moment the question is asked (`queryTimestamp`): the end of the
  conversation, or for knowledge-update questions, the update the dataset cites. Each recall asks
  for up to 100 results within 8,000 tokens of evidence.

## Files

| File | What it does |
|---|---|
| `run.py` | the whole run, one function per step |
| `dataset.py` | loads BEAM and turns each conversation into messages and questions |
| `past_api.py` | the past.dev API calls |
| `run_fibonna.py` | the same run against Fibonna ([how](#running-against-fibonna)) |
| `fibonna_api.py` | the Fibonna API calls |
| `tests/` | offline tests of the Fibonna adapter and runner |
| `exabase_prompts.py` | ExaBase's answer and judge prompts |
| `llm.py` | the answer and judge model |
| `scoring.py` | scoring and the summary |
| `log.py` | logging |
| `verify.py` | checks a run's numbers and the prompts |
| `NOTICE.md` | credits for the BEAM dataset and ExaBase's prompts |

## Attribution and license

The answer and judge prompts are ExaBase's work, reproduced unchanged from its
[published BEAM adapter](https://fabric.so/p/beam-3VjBcqEVRofZyA5CeazeX).
Past Corp's original harness code is covered by the repository's [MIT license](../LICENSE).
The ExaBase-derived prompts are excluded from that license; see [NOTICE.md](NOTICE.md)
for their attribution and the BEAM dataset's separate terms.
