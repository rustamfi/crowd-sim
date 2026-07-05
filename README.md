# SF Crowd Voting Simulator

Simulate **30 demographically representative San Francisco residents** voting on a ballot measure, then watch the results roll in. Each resident ("agent") is built from real U.S. Census microdata, given a Big Five (OCEAN) personality, and asked to vote by an LLM reasoning model of your choice.

By default the crowd votes on a measure to **cap food delivery app fees at 15%** — but you can edit the ballot question to ask them anything.

> Results are **non-deterministic** on purpose: the same 30 people can vote differently across runs because the model reasons freshly each time. Changing the *seed* selects a different 30-person sample from the Census data.

---

## Table of Contents

- [Quick Start](#quick-start)
- [User Manual (Web UI)](#user-manual-web-ui)
  - [The screen at a glance](#the-screen-at-a-glance)
  - [Your first simulation in 5 steps](#your-first-simulation-in-5-steps)
  - [Control reference](#control-reference)
  - [Reading an agent card](#reading-an-agent-card)
  - [Reading the Results Dashboard](#reading-the-results-dashboard)
  - [Campaign mode](#campaign-mode)
  - [Common workflows](#common-workflows)
  - [Tips & gotchas](#tips--gotchas)
- [Configuration Reference](#configuration-reference)
- [Command Line (advanced)](#command-line-advanced)
- [How It Works](#how-it-works)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
#    then edit .env — see "Configuration Reference" below

# 3. Launch the app
python src/app.py
```

Open **http://localhost:8000** in your browser.

You need two free API keys (details in [Configuration Reference](#configuration-reference)):

| Key | Get it from | Needed for |
|-----|-------------|------------|
| `CENSUS_API_KEY` | [census.gov](https://api.census.gov/data/key_signup.html) | Downloading Census data (first run only) |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) | Voting and agent memory (each vote costs a little money) |

---

## User Manual (Web UI)

Everything below happens in the browser at **http://localhost:8000** — no command line needed once the server is running.

### The screen at a glance

From top to bottom, the page is laid out like this:

```
┌────────────────────────────────────────────────────────────┐
│  SF Crowd Voting Simulator                        (header)  │
├────────────────────────────────────────────────────────────┤
│  ▸ Configuration            "No agents loaded"  (controls)  │   ← Model · Seed · Agent memory · Generate
├────────────────────────────────────────────────────────────┤
│  Ballot Question  [ editable text box ]   Reset to default  │
├────────────────────────────────────────────────────────────┤
│  [ Single Run ]  [ Campaign ]                    (mode tabs)│
├────────────────────────────────────────────────────────────┤
│                                                             │
│   Single Run: Run Vote / Re-vote                            │
│   Agent Population  (grid of 30 cards, All/Yes/No filter)   │
│   Results Dashboard (tally · charts · themes · highlight)   │
│                                                             │
│   Campaign: factor + seeds + Run Campaign                   │
│   Campaign Results (per-arm stats · distribution histogram) │
└────────────────────────────────────────────────────────────┘
```

- **Configuration**, **Ballot Question**, and the **mode tabs** are shared by both modes.
- The **Configuration** panel is collapsed by default — click it to expand.
- A **status line** on the right of the controls bar tells you the current state ("No agents loaded", "30 agents", which model produced the votes, etc.). Errors appear as a dismissible red banner just below.

### Your first simulation in 5 steps

1. **Open Configuration**, confirm a **Model** is selected, and leave **Seed** at `42`.
2. Click **Generate Population**. A grid of 30 resident cards appears.
3. (Optional) Edit the **Ballot Question** — or leave the default food-delivery-fee-cap measure.
4. Click **Run Vote** and wait ~30 seconds. Each card flips to show a **Yes** or **No** badge and a one-line reason.
5. Scroll down to the **Results Dashboard** to see the tally, charts, and themes. Click **Re-vote** to run the same crowd again and see whether the outcome holds.

That's the whole core loop: **Generate → (edit question) → Vote → read results**.

### Control reference

**Configuration panel**

| Control | What it does |
|---------|--------------|
| **Model** | The LLM that casts each vote (and writes agent memories). Choose from the dropdown; the list is curated in `src/llm.py`. Changing it affects the *next* vote or population you run. |
| **Seed** | A number (1–9999) that picks *which* 30 residents are drawn from the Census data. The same seed always yields the same 30 people; a new seed gives a fresh crowd. |
| **Agent memory** (checkbox) | When on, each generated agent gets **3 short, LLM-written past food-delivery experiences** that color their vote. Slower to generate (one extra LLM call per agent) and requires your OpenRouter key. Off = faster, no LLM used during generation. |
| **Generate Population** | Draws and displays 30 new agents using the current Model / Seed / Memory settings. **This clears any existing votes** — you start clean. |

**Ballot Question box**

| Control | What it does |
|---------|--------------|
| **Question text area** | The measure the crowd votes on. Pre-filled with the default fee-cap question; edit it to ask anything. |
| **Reset to default** | Restores the original question wording. |

**Single Run buttons**

| Control | What it does |
|---------|--------------|
| **Run Vote** | Sends all 30 agents to the model concurrently (~30s). Disabled until you've generated a population. |
| **Re-vote** | Discards cached votes and runs the same crowd again for a fresh, non-deterministic result. Appears after the first vote. |

### Reading an agent card

Each of the 30 cards shows one synthetic resident:

- **Name, age, neighborhood** — identity drawn from Census microdata.
- **Occupation, household income bracket, home tenure** (owner/renter).
- **OCEAN mini-chart** — five bars for Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism.
- **Agent memory** (if enabled) — an expandable list of that person's 3 past delivery experiences.
- **Vote badge + reason** — appears after voting: a green **Yes**, red **No**, or grey **null** (if the model failed to return a clean answer), plus the one-sentence rationale. A colored stripe on the card edge matches the vote.

Use the **All / Yes / No** filter above the grid to show only agents who voted a certain way.

### Reading the Results Dashboard

After a vote, the dashboard summarizes the outcome:

- **Vote tally cards** — Yes count, No count, and total valid votes, with percentages.
- **Vote Distribution** — a donut chart of Yes vs. No.
- **Vote by Income Bracket** — a grouped bar chart showing how support splits across household-income tiers (useful for spotting who a measure helps or hurts).
- **Themes** — the most common reasons *for* and *against*, grouped automatically by keyword.
- **Most interesting response** — one highlighted vote shown with full persona context.

### Campaign mode

Click the **Campaign** tab to run *many* populations at once and see the **distribution** of outcomes instead of a single result. This answers two kinds of question:

- *"How much does the answer wobble from crowd to crowd?"* (robustness)
- *"Does changing X move the vote?"* (comparison)

**Controls**

| Control | What it does |
|---------|--------------|
| **Compare factor** | What to vary across *arms* (side-by-side variants): **None** (one arm, robustness only), **Question wording**, **Agent memory** (on vs. off), or **Model** (several models head-to-head). When you pick a factor, a config area appears for defining each level. |
| **Populations per arm** | How many seeds (2–20) each arm runs. Every arm runs the **same seed set**, so comparisons are apples-to-apples. The hint line shows the total vote count (e.g. `2 arms × 5 × 30 = 300 votes`). |
| **Run Campaign** | Kicks off the run. A progress bar tracks it seed-by-seed. |

**Results** appear as one **stat card per arm** plus an overlaid **histogram** of Yes-vote % across seeds — a wide spread means the outcome is sensitive to which crowd you drew; well-separated arms mean the factor genuinely moved the vote.

> **Campaign runs are isolated.** They vote against throwaway state, so running a campaign never disturbs your Single Run population or results on the other tab.

### Common workflows

- **Test one crowd's stability** — Generate → Run Vote → **Re-vote** a few times. If the tally barely moves, the result is robust for that crowd.
- **See if the crowd matters** — Campaign tab, factor **None**, 10 populations. A tight histogram means the answer is crowd-independent.
- **Compare two phrasings** — Campaign tab, factor **Question wording**, enter both versions, and compare the two histograms.
- **Compare models** — Campaign tab, factor **Model**, select several, and see which models lean Yes vs. No.
- **Study the effect of memory** — Campaign tab, factor **Agent memory**, to see how lived "experience" shifts the vote.

### Tips & gotchas

- **Votes cost money.** Each vote is a real LLM call through OpenRouter. Campaign runs multiply that (arms × seeds × 30) — check the vote-count hint before launching a big one.
- **Question text is part of the cache.** Re-voting with the *exact same* question reuses cached votes; any edit triggers a fresh vote. Use **Re-vote** to force a re-run without changing the wording.
- **Generating wipes votes.** Clicking **Generate Population** always clears the previous results — do it deliberately.
- **Same seed = same people.** Only the *voting* is non-deterministic; the population is fully reproducible from its seed.

---

## Configuration Reference

Set these in your `.env` file (copy from `.env.example`):

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `CENSUS_API_KEY` | First run only | — | Downloads Census PUMS microdata. After the first fetch it's cached to `data/pums_sf.csv` and never needed again. |
| `OPENROUTER_API_KEY` | For voting & memory | — | Authenticates LLM calls via [OpenRouter](https://openrouter.ai/keys). **Real votes cost real money.** |
| `OPENROUTER_MODEL` | No | first model in `src/llm.py` | Default model id (must be a slug listed in `llm.MODELS`, e.g. `anthropic/claude-sonnet-5`). Leave blank to use the first registry entry. |
| `MAX_CONCURRENCY` | No | `10` | Number of parallel LLM calls during voting/memory generation. |
| `PORT` | No | `8000` | Server port (set automatically on Railway). |

**Changing the selectable models:** edit the `MODELS` list in `src/llm.py`. Keep strong roleplay models near the top — the personas depend on it. Model ids are OpenRouter slugs (`provider/model-name`).

---

## Command Line (advanced)

The three pipeline stages can also be run directly, without the web UI:

```bash
# 1. Generate 30 personas from Census data (writes results/agents.json)
python src/generate_population.py --seed 42

#    ...with agent memory (requires OPENROUTER_API_KEY; --model is optional)
python src/generate_population.py --seed 42 --memory --model anthropic/claude-sonnet-5

# 2. Have all agents vote (writes results/responses.jsonl; skips already-cached agents)
python src/run_scenario.py --model anthropic/claude-sonnet-5

# 3. Group reasons into themes and write results/summary.md
python src/analyze.py
```

To force a completely fresh vote, delete the cache first:

```bash
rm results/responses.jsonl
python src/run_scenario.py
```

Run the tests with:

```bash
python -m pytest src/tests/
```

---

## How It Works

A three-stage file-based pipeline, wrapped by a FastAPI web app:

```
generate_population.py  →  results/agents.json      (30 personas from Census data)
run_scenario.py         →  results/responses.jsonl  (concurrent LLM votes)
analyze.py              →  results/summary.md        (themed summary)
```

**Data source.** U.S. Census **ACS 2022 5-Year PUMS** microdata for San Francisco County. Because PUMS provides individual-level records (with person-weights), the app can build a synthetic person whose age, income, occupation, and housing status are all internally consistent — something aggregate summary tables can't do.

**Representative sampling.** The 30 agents are drawn to match SF's actual adult population on sex, race/ethnicity, home tenure, and household-income bracket (targets computed from the PUMS weights themselves). A hard cap of **4 software engineers** keeps any single sample from being dominated by tech.

**OCEAN personality.** Each agent's Big Five scores are **derived from their demographics** (age, household income, sex, tenure, occupation category) plus a small per-agent random wobble — never assigned at random. Demographics are the dominant signal; the noise just keeps two similar people from being identical. See the `_derive_ocean` docstring in `src/generate_population.py` for the exact coefficients.

**Voting.** Each agent's persona becomes the model's `system` message and the ballot question is the `user` message. The model returns strict JSON (`vote` + `reason`). All calls go through `src/llm.py` (OpenRouter). Votes are cached per agent so re-analysis is free; **Re-vote** clears the cache.

**Reproducibility.** Population sampling and per-agent attributes are fully seeded — the same seed always yields the same 30 people. Only the *voting* is non-deterministic (the model reasons freshly each time), which is what makes **Re-vote** and **Campaign** mode meaningful.

For the full requirement list and data-model details, see `specs/prd.md` and `specs/architecture.md`.

---

## Deployment

The repo ships ready for **Railway**:

- `railway.json` — Nixpacks builder, start command, health check on `/health`
- `Procfile` — `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
- `.python-version` — pins the Python runtime

**Steps:**
1. Connect your GitHub repo to Railway.
2. Set environment variables: `OPENROUTER_API_KEY`, `CENSUS_API_KEY` (and optionally `OPENROUTER_MODEL`, `MAX_CONCURRENCY`).
3. Railway builds and deploys; it probes `/health` before routing traffic. `PORT` is set automatically.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "OPENROUTER_API_KEY is not set" when voting | Add the key to `.env` (or the server's environment) and restart. |
| Census download fails on first run | Check `CENSUS_API_KEY`. Once `data/pums_sf.csv` exists, no key or network is needed. |
| Model dropdown is empty / "Loading models…" | The app couldn't reach `/api/config`; confirm the server is running and reachable. |
| Votes seem stale after changing the question | The exact question text is part of the cache key — a changed question re-votes; identical text reuses cached votes. Use **Re-vote** to force a fresh run. |
| Want a different crowd | Change the **Seed** and click **Generate Population**. |

---

## License

For educational and research purposes. Census PUMS data is public domain (U.S. government work).
```

