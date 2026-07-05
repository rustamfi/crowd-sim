"""
SF Crowd Voting Simulator — FastAPI Web Application
====================================================
Serves the API endpoints and static HTML frontend.
Imports core logic from generate_population, run_scenario, and analyze.

REQ-033 through REQ-038.

Bind to 0.0.0.0:$PORT (Railway sets PORT; default 8000 for local dev).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

load_dotenv(Path(__file__).parent.parent / ".env")
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
STATIC_DIR = ROOT / "static"
AGENTS_PATH = RESULTS_DIR / "agents.json"
RESPONSES_PATH = RESULTS_DIR / "responses.jsonl"

# ---------------------------------------------------------------------------
# Import core modules
# ---------------------------------------------------------------------------
# These imports must succeed before any request is served.
# All three modules are free of top-level side effects.
sys.path.insert(0, str(Path(__file__).parent))

import llm                                                  # noqa: E402
from generate_population import generate                    # noqa: E402
from run_scenario import run_votes, DEFAULT_QUESTION         # noqa: E402
from analyze import (                     # noqa: E402
    analyze,
    _load_agents,
    _load_responses,
)


def _llm_client():
    """Build the OpenRouter client if a key is configured, else None (analyze falls back)."""
    return llm.build_client()

# ---------------------------------------------------------------------------
# FastAPI app — REQ-033
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SF Crowd Voting Simulator",
    description=(
        "Simulates 30 demographically representative SF residents voting on "
        "a food delivery fee cap ballot measure."
    ),
    version="1.0.0",
)

# Serve static files (index.html and any future assets) — REQ-036
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes — REQ-034
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index():
    """Serve the HTML frontend. REQ-033, REQ-035."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Ensure static/index.html exists.",
        )
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/health", include_in_schema=False)
async def health():
    """Lightweight liveness probe for Railway's health check. REQ-038."""
    return {"status": "ok"}


@app.post("/api/generate")
async def api_generate(
    seed: int = Query(default=42, description="Random seed"),
    memory: bool = Query(
        default=False,
        description="Give each agent 3 LLM-generated past delivery-app experiences",
    ),
    model: str = Query(
        default=None,
        description="OpenRouter model for agent memory (defaults to the server default)",
    ),
):
    """
    Trigger population generation. Calls generate() and writes results/agents.json.
    Returns the list of 30 agent objects. REQ-034.

    When memory=true, each agent additionally gets a "delivery_experiences" list
    that influences their vote (requires OPENROUTER_API_KEY). ``model`` selects the
    OpenRouter model used for that step.
    """
    try:
        agents = generate(seed=seed, use_memory=memory, memory_model=model)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Population generation failed: {exc}",
        ) from exc

    # Persist to disk and clear stale responses from previous population
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(AGENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(agents, f, indent=2, ensure_ascii=False)
    if RESPONSES_PATH.exists():
        RESPONSES_PATH.unlink()
    summary_path = RESULTS_DIR / "summary.md"
    if summary_path.exists():
        summary_path.unlink()

    return {"status": "ok", "count": len(agents), "agents": agents}


@app.get("/api/config")
async def api_config():
    """
    Return client configuration: the default ballot question and the set of
    selectable OpenRouter models (id + label) plus the default model id.
    """
    return {
        "status": "ok",
        "default_question": DEFAULT_QUESTION,
        "models": llm.MODELS,
        "default_model": llm.DEFAULT_MODEL,
    }


@app.post("/api/vote")
async def api_vote(
    fresh: bool = Query(default=False, description="Clear cached votes and re-run all agents"),
    model: str = Query(
        default=None,
        description="OpenRouter model to vote with (defaults to the server default)",
    ),
    question: str = Body(
        default=None,
        embed=True,
        description="Ballot question to ask each agent (defaults to the SF fee cap measure).",
    ),
):
    """
    Trigger voting scenario for all agents. Requires OPENROUTER_API_KEY. REQ-034, REQ-037.
    Pass ?fresh=true to discard cached votes and re-run (non-deterministic results).
    ``model`` selects the OpenRouter reasoning model. An optional JSON body
    {"question": "..."} overrides the default ballot question; cached votes are
    reused only when the same question is asked again.
    """
    # REQ-037: Check API key before doing any work
    if not llm.api_key():
        raise HTTPException(
            status_code=400,
            detail=(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Set it on the server before calling this endpoint."
            ),
        )

    if not AGENTS_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="No agents found. Call POST /api/generate first.",
        )

    # Clear cached responses for a fresh re-vote
    if fresh and RESPONSES_PATH.exists():
        RESPONSES_PATH.unlink()

    with open(AGENTS_PATH, encoding="utf-8") as f:
        agents = json.load(f)

    question = (question or "").strip() or DEFAULT_QUESTION

    try:
        vote_results = run_votes(agents, question=question, model=model)
    except EnvironmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Voting scenario failed: {exc}",
        ) from exc

    # Run analysis inline to produce summary for the frontend
    try:
        summary = analyze(agents, vote_results, client=_llm_client())
    except Exception:
        summary = {}

    yes_count = sum(1 for r in vote_results if r.get("vote") == "Yes")
    no_count = sum(1 for r in vote_results if r.get("vote") == "No")
    null_count = sum(1 for r in vote_results if r.get("vote") is None)
    total_valid = yes_count + no_count

    return {
        "status": "ok",
        "question": question,
        "model": llm.resolve_model(model),
        "yes_count": yes_count,
        "no_count": no_count,
        "null_count": null_count,
        "pct_yes": round(yes_count / total_valid * 100, 1) if total_valid > 0 else 0.0,
        "yes_pct": round(yes_count / total_valid * 100, 1) if total_valid > 0 else 0.0,
        "no_pct": round(no_count / total_valid * 100, 1) if total_valid > 0 else 0.0,
        "results": vote_results,
        "summary": summary,
    }


@app.post("/api/campaign/run")
async def api_campaign_run(
    seed: int = Query(description="Population seed for this campaign run (1..20)"),
    model: str = Query(
        default=None,
        description="OpenRouter model to vote with (defaults to the server default)",
    ),
    memory: bool = Query(
        default=False,
        description="Give each agent 3 LLM-generated past delivery-app experiences",
    ),
    question: str = Body(
        default=None,
        embed=True,
        description="Ballot question to ask each agent (defaults to the SF fee cap measure).",
    ),
):
    """
    Run a SINGLE isolated seed of a simulation campaign: generate a population,
    have all 30 agents vote, and return that run's Yes/No tally. Requires
    OPENROUTER_API_KEY. Unlike /api/vote, this never touches the shared single-run
    state (results/agents.json, responses.jsonl, summary.md) — votes go to a
    throwaway temp file — so a campaign leaves the on-screen single run intact.

    The frontend calls this once per seed (1..N) and aggregates the returned
    ``pct_yes`` values into a distribution histogram.
    """
    # Mirror /api/vote: check API key before doing any work
    if not llm.api_key():
        raise HTTPException(
            status_code=400,
            detail=(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Set it on the server before calling this endpoint."
            ),
        )

    # Server-side cap so a crafted request can't launch an unbounded run
    if not 1 <= seed <= 20:
        raise HTTPException(
            status_code=400,
            detail="seed must be between 1 and 20.",
        )

    question = (question or "").strip() or DEFAULT_QUESTION

    try:
        agents = generate(seed=seed, use_memory=memory, memory_model=model)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Population generation failed: {exc}",
        ) from exc

    # Vote against a throwaway file so the shared responses.jsonl is untouched and
    # the vote cache never collides across seeds (agent ids are always 1..30).
    fd, tmp_name = tempfile.mkstemp(suffix=".jsonl", prefix="campaign_")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        vote_results = run_votes(
            agents, question=question, model=model, output_path=tmp_path
        )
    except EnvironmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Voting scenario failed: {exc}",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    yes_count = sum(1 for r in vote_results if r.get("vote") == "Yes")
    no_count = sum(1 for r in vote_results if r.get("vote") == "No")
    null_count = sum(1 for r in vote_results if r.get("vote") is None)
    total_valid = yes_count + no_count

    # Per-agent household income + vote so the frontend can pool an income-bracket
    # breakdown across seeds/arms. Join on id (household income lives on the agent,
    # the vote on the result), same pattern as analyze._join. Kept lightweight — no
    # reasons/personas — and the bracket definitions stay solely in the frontend.
    vote_by_id = {int(r["id"]): r.get("vote") for r in vote_results}
    voters = [
        {"household_income": a.get("household_income"), "vote": vote_by_id.get(int(a["id"]))}
        for a in agents
    ]

    return {
        "status": "ok",
        "seed": seed,
        "model": llm.resolve_model(model),
        "yes_count": yes_count,
        "no_count": no_count,
        "null_count": null_count,
        "pct_yes": round(yes_count / total_valid * 100, 1) if total_valid > 0 else 0.0,
        "voters": voters,
    }


@app.get("/api/agents")
async def api_get_agents():
    """
    Return the current agents from results/agents.json. REQ-034.
    """
    if not AGENTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No agents found. Call POST /api/generate first.",
        )
    try:
        agents = _load_agents(AGENTS_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load agents: {exc}",
        ) from exc
    return {"status": "ok", "count": len(agents), "agents": agents}


@app.get("/api/results")
async def api_get_results():
    """
    Return voting results and summary data. Runs analyze() over current files. REQ-034.
    """
    if not AGENTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No agents found. Call POST /api/generate first.",
        )
    if not RESPONSES_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No responses found. Call POST /api/vote first.",
        )

    try:
        agents = _load_agents(AGENTS_PATH)
        responses = _load_responses(RESPONSES_PATH)
        summary = analyze(agents, responses, client=_llm_client())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    # Build vote_results list for the frontend (id, vote, reason per agent)
    response_map = {int(r["id"]): r for r in responses}
    vote_results = [
        {
            "id": int(a["id"]),
            "name": a["name"],
            "vote": response_map.get(int(a["id"]), {}).get("vote"),
            "reason": response_map.get(int(a["id"]), {}).get("reason"),
        }
        for a in agents
    ]

    # Report the model(s) that produced these cached votes so the UI can show it.
    models_used = sorted({r.get("model") for r in responses if r.get("model")})
    model = models_used[0] if len(models_used) == 1 else (models_used or None)

    yes_count = summary.get("yes_count", 0)
    no_count = summary.get("no_count", 0)
    total_valid = yes_count + no_count

    return {
        "status": "ok",
        "model": model,
        "results": vote_results,
        "summary": {
            **summary,
            "yes_pct": round(yes_count / total_valid * 100, 1) if total_valid > 0 else 0.0,
            "no_pct": round(no_count / total_valid * 100, 1) if total_valid > 0 else 0.0,
        },
        **summary,
    }


# ---------------------------------------------------------------------------
# Entrypoint — REQ-032, REQ-038
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
