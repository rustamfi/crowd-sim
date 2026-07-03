# SF Crowd Voting Simulator

Simulates 30 demographically representative San Francisco residents voting on a ballot measure to cap food delivery app fees at 15%. Each persona is grounded in real U.S. Census microdata, enriched with Big Five (OCEAN) personality scores derived from demographic correlates, and votes via OpenAI's o4-mini reasoning model.

The simulation produces non-deterministic outcomes: the same 30 people can vote differently across runs because o4-mini performs internal chain-of-thought reasoning that varies each time. Changing the seed selects a different 30-person sample from the Census data.

## Table of Contents

- [Data Source](#data-source)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Pipeline](#pipeline)
- [OCEAN Personality Derivation](#ocean-personality-derivation)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [File Structure](#file-structure)
- [Agent List and Voting Results](#agent-list-and-voting-results)
- [Deployment](#deployment)

## Data Source

**American Community Survey (ACS) 2022 5-Year Public Use Microdata Sample (PUMS)** from the U.S. Census Bureau.

- **Endpoint:** `https://api.census.gov/data/2022/acs/acs5/pums`
- **Geography:** San Francisco County, California (state FIPS 06, PUMAs 07507-07514 under 2020 PUMA definitions)
- **Fields fetched:** `AGEP` (age), `OCCP` (occupation code), `PINCP` (personal income), `HINCP` (household income), `TEN` (tenure), `RAC1P` (race), `HISP` (Hispanic origin), `SEX`, `PWGTP` (person weight), `PUMA20`
- **Total SF records:** ~8,500 individual-level person records
- **API key:** Free, required. Register at [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)

PUMS provides individual-level microdata with person weights, enabling weighted sampling that reflects the actual population distribution of San Francisco. This is fundamentally different from summary tables, which only provide aggregate counts and cannot produce correlated attributes for a single synthetic person.

The raw Census response is cached to `data/pums_sf.csv` on first fetch. All subsequent runs use the cached file with no network call.

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd crowd-sim

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

| Key | Source | Purpose |
|-----|--------|---------|
| `CENSUS_API_KEY` | [census.gov](https://api.census.gov/data/key_signup.html) (free, instant) | Fetching PUMS microdata (first run only) |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | o4-mini reasoning model for vote simulation |

## How to Run

### Web App (primary)

```bash
python src/app.py
```

Opens at [http://localhost:8000](http://localhost:8000). The web interface provides:

- **Generate Population** -- Create 30 agents from Census data with a configurable seed
- **Run Vote** -- Trigger o4-mini voting for all 30 agents (concurrent, ~30s)
- **Re-vote** -- Re-run voting for the same population with fresh non-deterministic results
- **Results Dashboard** -- Vote tally chart, themed reasons, most interesting response, sortable agent table

Existing `results/agents.json` and `results/responses.jsonl` are loaded automatically on page load.

### CLI (alternative)

```bash
# Step 1: Generate 30 personas from Census data
python src/generate_population.py [--seed 42] [--output results/agents.json]

# Step 2: Run voting scenario via OpenAI o4-mini
python src/run_scenario.py [--agents results/agents.json] [--output results/responses.jsonl]

# Step 3: Analyze results and produce summary
python src/analyze.py [--agents results/agents.json] [--responses results/responses.jsonl] [--output results/summary.md]
```

Each step is independently re-runnable. Step 2 skips agents whose responses are already cached in the JSONL file.

## Pipeline

### Step 1: Population Generation (`src/generate_population.py`)

1. **Fetch** ACS 2022 5-year PUMS data for all of California (PUMA20 variable), cache to `data/pums_sf.csv`
2. **Filter** to SF PUMAs (07507-07514) and adults (age >= 18) -- yields ~7,500 eligible records
3. **Weighted-sample** 30 records using `random.choices()` with `PWGTP` person weights and a fixed seed (default 42)
4. **Enforce** max 4 software engineers (over-sample pool of 90, then cap)
5. **Map** each record's PUMA to an SF neighborhood via hardcoded lookup (20 neighborhoods across 8 PUMAs)
6. **Map** OCCP occupation codes to human-readable names (200+ specific codes + range-based fallback for 22 broad categories)
7. **Assign** culturally plausible names from ethnicity-stratified lists (keyed by RAC1P/HISP + SEX, 30 first names and 20 last names per group)
8. **Derive** OCEAN personality scores from demographic adjustments + seeded noise (see table below)
9. **Generate** a 2-sentence behavioral profile per agent (deterministic, no LLM)
10. **Write** 30 agent objects to `results/agents.json`

### Step 2: Voting Scenario (`src/run_scenario.py`)

1. **Load** agents from `results/agents.json`
2. **Check** for cached responses in `results/responses.jsonl`, skip completed agents
3. **Construct** per-agent prompts:
   - **Developer prompt** (o4-mini uses `developer` role, not `system`): full persona card with name, age, occupation, neighborhood, income, tenure, OCEAN scores, and behavioral profile
   - **User prompt**: the exact ballot question about capping food delivery fees at 15%
4. **Call** OpenAI o4-mini with `reasoning_effort="medium"` and `response_format={"type": "json_object"}` -- **concurrently** (default 10 parallel requests via `MAX_CONCURRENCY`)
5. **Parse** JSON response for `vote` ("Yes"/"No") and `reason` (one sentence)
6. **Retry** once on parse failure; record null vote on second failure
7. **Append** each response to `results/responses.jsonl` (thread-safe with write lock)

The prompt never mentions Prop F, Proposition F, 2021, or any real-world prior vote outcome.

### Step 3: Analysis (`src/analyze.py`)

1. **Join** agents and responses on `id`
2. **Compute** vote tallies (Yes/No/null counts and percentages)
3. **Group** reasons by theme using keyword/substring matching (no LLM call) -- 6 Yes themes and 6 No themes documented in the module docstring
4. **Select** the most interesting response (longest reason text) with full agent context
5. **Write** `results/summary.md` with: vote tally, top Yes/No reasons, most interesting response as blockquote, and full agent table

## OCEAN Personality Derivation

Each agent receives Big Five personality scores on a 1-10 scale. Demographic/contextual factors are the **primary signal**; per-agent noise adds minor individual variation.

### Algorithm

```
score = 5.5 (base)
      + sum of applicable demographic adjustments (stack additively)
      + random.Random(agent_id).gauss(0, 0.4) per trait
      → clamp to [1.0, 10.0], round to 1 decimal
```

### Adjustment Table

O = Openness, C = Conscientiousness, E = Extraversion, A = Agreeableness, N = Neuroticism

| Condition | O | C | E | A | N | Rationale |
|-----------|--:|--:|--:|--:|--:|-----------|
| Age < 30 | +0.7 | -0.5 | +0.6 | -0.3 | +0.5 | Youth: higher openness/extraversion, lower conscientiousness |
| Age 30-50 | 0 | +0.3 | 0 | +0.2 | 0 | Mid-life: settling, more agreeable/conscientious |
| Age > 50 | -0.4 | +0.6 | -0.4 | +0.5 | -0.5 | Maturity: higher C/A, lower N/O/E |
| Female | +0.2 | +0.2 | +0.2 | +0.5 | +0.5 | Population-level tendencies in A and N |
| Male | 0 | 0 | 0 | -0.2 | -0.3 | Slight inverse of female adjustments |
| Income > $150k | +0.4 | +0.6 | +0.4 | 0 | -0.4 | High earners: conscientious, open, lower anxiety |
| Income < $30k | -0.2 | -0.2 | -0.2 | +0.2 | +0.6 | Economic stress: higher neuroticism |
| Homeowner | 0 | +0.4 | 0 | 0 | -0.3 | Stability correlates with conscientiousness |
| Renter | +0.2 | -0.2 | +0.2 | 0 | +0.2 | Flexibility, slightly higher openness |
| Tech/creative job | +0.8 | +0.3 | 0 | -0.2 | 0 | Tech/creative: high openness |
| Service/food job | 0 | 0 | +0.5 | +0.6 | +0.2 | Service: high extraversion/agreeableness |
| Healthcare job | 0 | +0.6 | +0.2 | +0.7 | 0 | Healthcare: high conscientiousness/agreeableness |
| Arts/media job | +1.0 | -0.3 | +0.3 | 0 | +0.3 | Artists: very high openness |
| Education job | +0.3 | +0.4 | +0.3 | +0.5 | 0 | Teachers: agreeable, conscientious |

Multiple conditions stack additively. A 25-year-old female tech worker renting in SF accumulates: Age<30 (+0.7 O) + Female (+0.2 O) + Income>$150k (+0.4 O) + Renter (+0.2 O) + Tech (+0.8 O) = **+2.3 Openness** before noise. The per-agent Gaussian noise (stddev 0.4) ensures that two agents with identical demographics still have meaningfully different personalities.

### Design Rationale

The adjustments are grounded in commonly cited population-level tendencies from personality psychology literature (e.g., age-related increases in conscientiousness and agreeableness, sex differences in agreeableness and neuroticism). They are scaled to be the dominant signal, with noise serving only to prevent agents within a demographic cluster from feeling identical. This is not a claim of empirical accuracy -- it provides enough behavioral texture for Claude to generate distinct, plausible persona responses.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CENSUS_API_KEY` | Yes (first run) | — | Free Census Bureau API key for PUMS data |
| `OPENAI_API_KEY` | Yes (voting) | — | OpenAI API key for o4-mini model |
| `MAX_CONCURRENCY` | No | `10` | Number of parallel API calls during voting |
| `PORT` | No | `8000` | Server port (set automatically by Railway) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the HTML frontend |
| `POST` | `/api/generate?seed=42` | Generate 30 agents from Census data |
| `POST` | `/api/vote` | Run voting for all agents (skips cached) |
| `POST` | `/api/vote?fresh=true` | Clear cache and re-vote all agents |
| `GET` | `/api/agents` | Return current agents from `results/agents.json` |
| `GET` | `/api/results` | Return votes, summary, and analysis |

## File Structure

```
crowd-sim/
  src/
    generate_population.py   # Census data -> 30 agent personas (1000+ lines)
    run_scenario.py          # Agent personas -> concurrent o4-mini votes
    analyze.py               # Votes -> themed summary report
    app.py                   # FastAPI web application
    __init__.py              # Package marker
  static/
    index.html               # Single-file frontend (HTML + CSS + vanilla JS)
  data/
    pums_sf.csv              # Cached Census PUMS data (~8,500 SF records)
  results/
    agents.json              # 30 generated agent personas
    responses.jsonl          # Cached vote responses (one JSON object per line)
    summary.md               # Analysis report with tallies and themes
  specs/
    prd.md                   # Product requirements (REQ-001 through REQ-038)
    architecture.md          # Architecture, data model, algorithms
  .env.example               # Environment variable template
  .gitignore                 # Excludes .env, __pycache__, .venv
  requirements.txt           # Python dependencies
  Procfile                   # Railway deployment
  README.md                  # This file
```

## Agent List and Voting Results

*Generated with seed 2. Results are non-deterministic -- re-running produces different vote outcomes for the same population.*

**Vote Tally: 26 Yes (86.7%) / 4 No (13.3%)**

**Top Yes Reasons:**
1. **Affordability / Cost of Living** (18 voters) -- "Capping fees protects residents on modest incomes from excessive delivery charges."
2. **Support for Small Restaurants** (8 voters) -- "Local restaurants are struggling with high commission costs that eat into thin margins."

**Top No Reasons:**
1. **Business Viability / App Quality** (4 voters) -- "A strict cap risks reducing service quality and delivery options."

**Most Interesting Response:**
> **Joseph Taylor**, age 44, Pacific Heights -- Management (Vote: **No**)
> "A strict 15% cap risks reducing service quality and delivery options by limiting platforms' ability to cover operating costs and pay drivers fairly."

### Full Agent Table

| # | Name | Age | Sex | Race/Ethnicity | Neighborhood | Occupation | HH Income | Tenure | O | C | E | A | N | Vote |
|---|------|-----|-----|----------------|-------------|------------|-----------|--------|---|---|---|---|---|------|
| 1 | Miguel Gutierrez | 35 | Male | Hispanic/Latino | Richmond | Computer/Math | >$120k | Renter | 6.5 | 5.7 | 5.8 | 4.9 | 5.1 | Yes |
| 2 | Walter Wang | 30 | Male | Asian | Richmond | Computer/Math | >$120k | Renter | 7.7 | 6.5 | 5.6 | 5.3 | 6.1 | Yes |
| 3 | Christine Kumar | 44 | Female | Asian | Nob Hill | Management | >$120k | Owner | 5.7 | 6.7 | 5.6 | 6.5 | 5.6 | Yes |
| 4 | Joseph Taylor | 44 | Male | White | Pacific Heights | Management | >$120k | Renter | 5.9 | 5.7 | 6.0 | 5.7 | 5.3 | No |
| 5 | Tina Lee | 24 | Female | Asian | North Beach | Aerospace Engineer | >$120k | Renter | 8.0 | 5.4 | 6.7 | 4.7 | 6.9 | Yes |
| 6 | Stanley Kumar | 23 | Male | Asian | Chinatown | Computer/Math | >$120k | Renter | 6.9 | 5.9 | 6.5 | 4.5 | 5.3 | Yes |
| 7 | Christopher Jones | 44 | Male | White | Sunset | Social and Human Service Assistant | >$120k | Renter | 5.6 | 5.7 | 6.4 | 5.8 | 5.2 | Yes |
| 8 | Joseph Jackson | 35 | Male | White | Nob Hill | Computer/Math | >$120k | Renter | 6.2 | 5.8 | 6.0 | 5.5 | 5.2 | Yes |
| 9 | Ashley Clark | 64 | Female | White | Cole Valley | Business/Finance | >$120k | Owner | 5.3 | 6.7 | 5.7 | 6.7 | 5.2 | Yes |
| 10 | Rafael Rodriguez | 47 | Male | Hispanic/Latino | Bayview | Computer/Math | >$120k | Renter | 7.3 | 6.6 | 6.0 | 5.0 | 4.8 | Yes |
| 11 | Nichelle Green | 62 | Female | Black or African American | South Beach | High School Teacher | $75k-$120k | Renter | 5.6 | 7.5 | 7.1 | 7.7 | 5.6 | Yes |
| 12 | Dorothy Anderson | 60 | Female | White | Sunset | Production | >$120k | Owner | 4.7 | 6.6 | 5.6 | 6.6 | 4.4 | Yes |
| 13 | Antonio Torres | 35 | Male | Hispanic/Latino | Sunset | Management | >$120k | Renter | 6.4 | 5.6 | 6.4 | 4.9 | 4.5 | No |
| 14 | Michael Clark | 20 | Male | White | Marina | Personal Care | >$120k | Owner | 5.7 | 5.5 | 6.6 | 5.0 | 6.5 | Yes |
| 15 | Amy Chen | 65 | Female | Asian | Mission | Healthcare Support | >$120k | Owner | 4.9 | 7.6 | 5.6 | 7.5 | 5.3 | Yes |
| 16 | Claudia Cruz | 79 | Female | Hispanic/Latino | Potrero Hill | Not in labor force | <$35k | Owner | 5.2 | 6.5 | 5.4 | 6.5 | 5.7 | Yes |
| 17 | Timothy Cheng | 76 | Male | Asian | Parkside | Protective Service | $75k-$120k | Renter | 5.3 | 5.9 | 5.2 | 6.2 | 4.7 | Yes |
| 18 | David Brown | 44 | Male | White | Visitacion Valley | Management | >$120k | Owner | 5.9 | 6.9 | 5.2 | 5.6 | 4.4 | Yes |
| 19 | Connie Wang | 83 | Female | Asian | SoMa | Not in labor force | <$35k | Renter | 5.5 | 5.9 | 5.3 | 6.9 | 5.9 | Yes |
| 20 | Cynthia Jones | 49 | Female | White | Visitacion Valley | Management | >$120k | Owner | 6.0 | 7.5 | 5.8 | 5.9 | 5.4 | Yes |
| 21 | Daniel Cheng | 78 | Male | Asian | Financial District | Not in labor force | <$35k | Renter | 4.8 | 6.1 | 4.9 | 5.7 | 5.7 | Yes |
| 22 | Kathleen Jones | 34 | Female | White | Financial District | Not in labor force | <$35k | Renter | 5.0 | 6.3 | 5.4 | 5.6 | 5.9 | Yes |
| 23 | Hiro Yang | 40 | Male | Asian | Noe Valley | Management | >$120k | Owner | 5.8 | 7.1 | 6.0 | 5.5 | 4.3 | No |
| 24 | Kendrick Moore | 36 | Male | Black or African American | Cole Valley | Computer Systems Analyst | >$120k | Owner | 6.5 | 7.2 | 6.2 | 5.9 | 5.0 | No |
| 25 | Karen Chen | 70 | Female | Asian | Potrero Hill | Not in labor force | $75k-$120k | Renter | 5.3 | 6.1 | 5.2 | 7.0 | 5.7 | Yes |
| 26 | Eugene Park | 44 | Male | Asian | Visitacion Valley | Actor | >$120k | Owner | 5.7 | 5.3 | 6.2 | 5.7 | 5.4 | Yes |
| 27 | Winston Chan | 26 | Male | Asian | Russian Hill | Computer/Math | >$120k | Renter | 7.1 | 5.3 | 6.0 | 5.4 | 6.2 | Yes |
| 28 | Taylor Hassan | 58 | Female | Two or More Races | Richmond | Science | $35k-$75k | Owner | 5.0 | 6.9 | 4.8 | 6.4 | 6.1 | Yes |
| 29 | Paul Williams | 24 | Male | White | Chinatown | Other | >$120k | Owner | 6.3 | 4.7 | 5.8 | 5.4 | 6.1 | Yes |
| 30 | Emily Taylor | 22 | Female | White | North Beach | Janitor | <$35k | Renter | 7.0 | 4.0 | 6.5 | 6.2 | 7.2 | Yes |

### Population Demographics (seed 2)

| Category | Distribution |
|----------|-------------|
| **Race/Ethnicity** | 10 White, 10 Asian, 4 Hispanic/Latino, 2 Black, 1 Two or More Races, 3 Other |
| **Sex** | 16 Male, 14 Female |
| **Age Range** | 20-83 (median 44) |
| **Tenure** | 15 Owner, 15 Renter |
| **HH Income** | 5 below $35k, 1 at $35k-$75k, 3 at $75k-$120k, 21 above $120k |
| **Neighborhoods** | 20 distinct neighborhoods across all 8 SF PUMAs |
| **Occupations** | Management, Computer/Math, Healthcare, Education, Arts, Construction, Service, and more |

## Deployment

### Railway

The app includes a `Procfile` for Railway deployment:

```
web: uvicorn src.app:app --host 0.0.0.0 --port $PORT
```

**Steps:**
1. Connect your GitHub repo to Railway
2. Set environment variables: `OPENAI_API_KEY`, `CENSUS_API_KEY`, `MAX_CONCURRENCY` (optional)
3. Railway auto-detects the `Procfile` and deploys

The `PORT` environment variable is set automatically by Railway.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your keys

# Run the app
python src/app.py
# Server starts at http://localhost:8000
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | >=2.31 | Census Bureau API calls |
| `openai` | >=1.0 | OpenAI o4-mini API SDK |
| `fastapi` | >=0.104 | Web framework and API |
| `uvicorn` | >=0.24 | ASGI server |
| `python-multipart` | >=0.0.6 | FastAPI form data handling |
| `python-dotenv` | >=1.0 | `.env` file loading |

## Reproducibility

All randomness is seeded:

| Scope | Mechanism | Controls |
|-------|-----------|----------|
| Global sampling | `random.seed(seed)` | Which 30 PUMS records are selected |
| Per-agent attributes | `random.Random(agent_id)` | Neighborhood, name, OCEAN noise for each agent independently |
| LLM voting | Non-deterministic | o4-mini reasoning varies per call (by design -- enables Re-vote) |

Changing the seed changes the population. The same seed always produces the same 30 agents with the same attributes. Voting outcomes vary across runs because o4-mini's internal reasoning is non-deterministic -- this is intentional and models the inherent uncertainty in predicting human behavior.

## License

This project is for educational and research purposes. Census PUMS data is public domain (U.S. government work).
