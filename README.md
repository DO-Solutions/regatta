# Poseidon's Regatta

**Same question, same knowledge, three models.**

A dependency-free demo and eval harness that races **Kimi K2.6, Kimi K3, and Claude Opus 5**
on [DigitalOcean Serverless Inference](https://docs.digitalocean.com/products/genai-platform/)
against the same question, side by side. It then grades every answer with an LLM judge and
rolls the results into a scorecard with cost-at-scale projections.

Built for the DigitalOcean **Kimi K3 launch webinar** (2026-07-29). The demo segment and the
aggregate numbers shown there come from this code.

## Why the method looks the way it does

- **Identical prompts by construction.** Every model gets the byte-identical system prompt
  (`SYSMSG` in `race_server.py`) and, when retrieval is on, the same retrieved context.
  Retrieval runs once per question and is shared across all arms. Prompt and persona
  differences can move benchmark scores more than the model does. We measured one persona
  directive costing 9.5 faithfulness points, so the comparison holds those constant.
- **A third-family, open-source judge.** Verdicts come from `deepseek-v4-pro` (set
  `JUDGE_MODEL` to change it). No candidate is graded by its own vendor's model, so there is
  no self-preference bias to argue about.
- **Rate limits are not a quality signal.** The suite runner retries 429 and 5xx failures in
  later sweeps until they complete cleanly. Only clean completions are scored, and the
  excluded count is printed next to the results.
- **Degenerate output is a serving fault, not a score.** A response that is mostly one
  repeated character gets excluded and re-run, the same as a 429. We watched a model emit
  4,096 tokens of `!` under launch-day load. Scoring those as zeros would have buried a
  model that was otherwise answering in the 90s.
- **Reasoning tokens are counted.** Reasoning-first models spend completion tokens thinking
  before they answer, out of the same budget. The stats separate time-to-first-reasoning
  from time-to-first-answer, so a slower model that reasons is visible as such.

## Quick start

You need a DigitalOcean **model access key** with access to the models you want to race
(create one in the GenAI Platform console).

```bash
cp .env.example .env      # put your key in DO_KEY
./run_local.sh            # serves http://localhost:8130
```

Open the page, pick a question, **Ask all three**, then **Grade the answers**.

The bundled `evalset.csv` (57 questions with ground truth about the DO GenAI Platform) feeds
both the dropdown and the batch run:

```bash
set -a; . ./.env; set +a
python3 suite_runner.py --out suite_results.json
```

That prints per-model correctness, faithfulness, latency, p95, cost per answer, the
excluded-call count, and monthly cost projections at 10k, 100k, and 1M questions.

The **Podium** tab reads `suite_results_merged.json` and renders the aggregate: a
quality-against-cost scatter, per-model score and latency panels, a per-question heat strip,
and cost-at-scale bars. The bundled file holds the real webinar numbers, so the tab has
something to show before you run anything.

## Retrieval (optional)

Three modes, in precedence order:

1. **`RAG_CACHE=chunks.json`**, a precomputed `{question: [chunk, …]}` map. This is how the
   webinar numbers stay reproducible: every run sees byte-identical context.
2. **`RAG_URL` + `RAG_KEY`**, any live endpoint speaking `POST /search
   {"query", "collection", "top_k"}` and returning `{"results": [{"text": …}]}`.
3. **Neither**, a context-free comparison of the bare models (noted at startup).

## Notes

- `top_p`: Kimi K3 rejects any value other than `0.95` *if the parameter is sent*. Omitting
  it entirely is also fine. The other models get no `top_p`.
- Prices in `race_server.py MODELS` came from the DO API's own model records
  (`GET /v2/gen-ai/models`, the `pricing` field). Re-check them before quoting numbers.
- `OPUS_BACKEND=anthropic` plus `ANTHROPIC_KEY` routes the Opus arm through the Anthropic
  API instead of DO, which helps if your DO key does not include Anthropic models.
- `K3_BASE` (with `K3_MODEL_ID` and `K3_KEY`) routes the K3 arm to any OpenAI-compatible
  endpoint. The comparison is about models, not providers, and list price is unchanged.
- The judge budget is 2,000 tokens on purpose. Smaller budgets truncate on long answers and
  produce unparseable verdicts that look like scoring failures.

MIT licensed. PRs welcome.
