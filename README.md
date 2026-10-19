# Do large language models adapt to changing payoffs?

Code and data for the article **"Do large language models adapt to changing payoffs? Evidence from a repeated
two-player game with four payoff regimes"** (Ruxandra Stanomir and Sorin Anagnoste, Bucharest University of
Economic Studies), submitted to *Poslovna izvrsnost – Business Excellence*.

Four LLM versions play a repeated two-player game in which the payoff matrix changes every 25 rounds across four
regimes (two prisoner's dilemmas, a game in which mutual B is best for both players, and an anti-coordination game):

| Model | OpenRouter identifier | Provider |
|---|---|---|
| Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` | Anthropic |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | Google AI Studio |
| DeepSeek-V4.1-Flash | `deepseek/deepseek-v4.1-flash` | Together |
| GPT-4.1 mini | `openai/gpt-4.1-mini` | OpenAI |

Temperature 0.7, hidden reasoning disabled, provider pinned. Data collected on 13 October 2026:
**336 independent 100-round matches, 57,600 model decisions.**

## Design

- **Main arm:** every ordered pair of models, 4 phase orders (Williams-type balanced Latin square: 1-2-3-4, 2-4-1-3,
  4-3-2-1, 3-1-4-2), 3 independent repetitions (144 matches).
- **Label-swap arm:** the labels "Option A" and "Option B" are reversed (48 matches).
- **Decision-only arm:** no written reasoning; the answer is constrained to a JSON schema (48 matches).
- **Reference arm:** each model against always A, always B and tit-for-tat (96 matches).

Inference uses a match-level (cluster) bootstrap with 2,000 resamples and Holm-adjusted p-values. The exact prompts
are in [`prompts_appendix.md`](prompts_appendix.md).

## Repository contents

| File | Contents |
|---|---|
| `experiment.py` | The code that ran the experiment and builds every table of the article from the logs. |
| `prompts_appendix.md` | Exact system and user prompts for every arm, settings and parsing rule. |
| `results/results.xlsx` | All result tables of the article (one sheet per table) and the round-level data. |
| `results/rounds.jsonl` | One line per round (33,600 rounds). |
| `results/calls.jsonl.gz` | One line per API call: full prompt, raw model response, parsed decision, tokens, cost, model version and provider returned (gzip). |
| `results/decisions.csv.gz` | Every decision next to the raw response that produced it (gzip). |
| `results/settings.json` | Settings recorded at run time, code hash and the data-collection sessions. |
| `results/experiment_snapshot.py` | The code as it was when data collection started. It differs from `experiment.py` only in `REPS["main"]` (2 instead of 3): the third repetition of the main arm was added afterwards, as specified in advance, to meet the precision targets. |

## Reproducing the results

```bash
pip install openai pandas numpy openpyxl
gunzip -k results/calls.jsonl.gz results/decisions.csv.gz
python experiment.py analyze
```

`analyze` rebuilds `results/results.xlsx` from the logs; the tables are identical to those reported in the article.
On Windows without `gunzip`, decompress with 7-Zip or run:
`python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('results/calls.jsonl.gz','rb'), open('results/calls.jsonl','wb'))"`

Other commands:

```bash
python experiment.py test      # offline check of the whole pipeline with a simulated model (no API, no cost)
python experiment.py prompts   # writes the exact prompts (prompts_appendix.md)
python experiment.py check     # one real call per model (requires an API key)
python experiment.py run       # runs the experiment (requires an API key; resumable)
```

Running new matches requires an [OpenRouter](https://openrouter.ai) API key in the environment variable
`OPENROUTER_API_KEY`. No key is stored in this repository.
