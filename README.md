# Poseidon's Regatta

**Same question, same knowledge, three models.**

A dependency-free demo + eval harness that races **Kimi K2.6, Kimi K3, and Claude Opus 5**
on [DigitalOcean Serverless Inference](https://docs.digitalocean.com/products/genai-platform/)
against the same question, side by side, with each model's **reasoning streamed live** —
then grades every answer with an LLM judge and rolls the results up into an aggregate
scorecard with cost-at-scale projections.

Built for the DigitalOcean **Kimi K3 launch webinar** (2026-07-29). The pre-recorded demo
segment and the aggregate numbers shown there come from this exact code.

## Why the method looks the way it does

- **Identical prompts by construction.** Every model gets the byte-identical system prompt
  (`SYSMSG` in `race_server.py`) and, when retrieval is enabled, the *same* retrieved
  context — retrieval runs once per question and is shared across all arms. Persona and
  prompt differences can move benchmark scores more than model differences (we measured a
  persona directive costing ~9.5 faithfulness points), so the comparison eliminates them.
- **A third-family, open-source judge.** Verdicts come from `deepseek-v4-pro` (configurable
  via `JUDGE_MODEL`) — neither Kimi nor Claude is graded by its own vendor's model, so
  there's no self-preference bias to argue about.
- **Rate limits are not a quality signal.** The suite runner retries 429/5xx failures in
  later sweeps until they complete cleanly; only clean completions are scored, and the
  excluded count is reported next to the results.
- **Reasoning costs are visible.** Reasoning-first models spend completion tokens thinking
  before they answer. The race UI shows thinking in-flight, and the stats separate
  time-to-first-reasoning from time-to-first-answer, so "slower but shows its work" is
  visible rather than hidden in a latency mean.

## Quick start

You need a DigitalOcean **model access key** with access to the models you want to race
(create one in the GenAI Platform console).

```bash
cp .env.example .env      # put your key in DO_KEY
./run_local.sh            # serves http://localhost:8130
```

Open the page, pick a question, **Ask all three**, then **Grade the answers**.

The bundled `evalset.csv` (57 questions + ground truth about the DO GenAI Platform) powers
both the dropdown and the batch run:

```bash
set -a; . ./.env; set +a
python3 suite_runner.py --out suite_results.json
```

which prints per-model correctness / faithfulness / latency / p95 / $-per-answer, the
excluded-call count, and monthly cost projections at 10k / 100k / 1M questions.

## Retrieval (optional)

Three modes, in precedence order:

1. **`RAG_CACHE=chunks.json`** — a precomputed `{question: [chunk, …]}` map. This is how
   the webinar numbers stay reproducible: every run sees byte-identical context.
2. **`RAG_URL` + `RAG_KEY`** — any live endpoint speaking `POST /search
   {"query", "collection", "top_k"}` → `{"results": [{"text": …}]}`.
3. **Neither** — context-free comparison of the bare models (clearly noted at startup).

## Notes

- `top_p`: Kimi K3 rejects any value other than `0.95` *if the parameter is sent* —
  omitting it entirely is also fine. The other models get no `top_p`.
- Prices in `race_server.py MODELS` were pulled from the DO API's own model records
  (`GET /v2/gen-ai/models` → `pricing`); re-check them before quoting numbers.
- `OPUS_BACKEND=anthropic` + `ANTHROPIC_KEY` routes the Opus arm through the Anthropic API
  instead of DO — useful if your DO key doesn't include Anthropic models.
- `K3_BASE` (+ `K3_MODEL_ID`, `K3_KEY`) routes the K3 arm to any OpenAI-compatible endpoint.
  The comparison is about models, not providers — list price is unchanged either way.
- Judge budget is 2000 tokens deliberately: small judge budgets truncate on long answers
  and silently produce unparseable verdicts.

MIT licensed. PRs welcome.
