"""
SF Crowd Voting Simulator — Voting Scenario Runner
===================================================
Reads agents from results/agents.json, constructs per-agent persona prompts,
calls OpenAI o4-mini with reasoning_effort="medium", parses the JSON vote
response, and appends results to results/responses.jsonl.

REQ-014 through REQ-023.

Key o4-mini API differences (from architecture.md):
- Uses "developer" role, NOT "system"
- Uses reasoning_effort="medium", NOT temperature
- Supports response_format={"type": "json_object"}
- Does NOT mention Prop F, Proposition F, 2021, or any real-world prior outcome.
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "10"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DEFAULT_AGENTS = RESULTS_DIR / "agents.json"
DEFAULT_OUTPUT = RESULTS_DIR / "responses.jsonl"

MODEL = "o4-mini"

# ---------------------------------------------------------------------------
# Prompt templates — REQ-016, REQ-017, REQ-020
# These are module-level constants as required by REQ-016.
# NOTE: Must NOT mention Prop F, Proposition F, 2021, or any real prior vote.
# ---------------------------------------------------------------------------

DEVELOPER_PROMPT_TEMPLATE = """\
You are {name}, a {age}-year-old {occupation} living in {neighborhood}, San Francisco.
You earn ${income_annual:,}/year and {tenure_desc}.
{profile}

Personality (OCEAN 1-10): Openness {openness}, Conscientiousness {conscientiousness}, \
Extraversion {extraversion}, Agreeableness {agreeableness}, Neuroticism {neuroticism}.

When asked about local policy, respond as this person would, considering your life \
circumstances, personality, and values. Respond ONLY with valid JSON.\
"""

# REQ-017: exact default question — no additions or omissions
DEFAULT_QUESTION = (
    "San Francisco is voting on a measure that would cap food delivery app fees "
    "(DoorDash, Uber Eats) at 15%. As a resident, would you vote Yes or No? "
    "Give your single most important reason in one sentence."
)

# JSON response instruction appended to every question.
_JSON_INSTRUCTION = (
    'Respond in JSON: {"vote": "Yes" or "No", "reason": "your one sentence reason"}'
)


def _build_user_prompt(question: str) -> str:
    """Compose the full user prompt from a question plus the JSON instruction."""
    return question.strip() + "\n\n" + _JSON_INSTRUCTION


# Backward-compatible default prompt (used when no custom question is supplied).
USER_PROMPT = _build_user_prompt(DEFAULT_QUESTION)


def _build_developer_prompt(agent: dict) -> str:
    """Construct the per-agent developer prompt from the persona card."""
    tenure_desc = (
        "owns their home" if agent["tenure"] == "Owner" else "rents their home"
    )
    ocean = agent["ocean"]
    return DEVELOPER_PROMPT_TEMPLATE.format(
        name=agent["name"],
        age=agent["age"],
        occupation=agent["occupation"],
        neighborhood=agent["neighborhood"],
        income_annual=max(0, agent["income_annual"]),
        tenure_desc=tenure_desc,
        profile=agent["profile"],
        openness=ocean["openness"],
        conscientiousness=ocean["conscientiousness"],
        extraversion=ocean["extraversion"],
        agreeableness=ocean["agreeableness"],
        neuroticism=ocean["neuroticism"],
    )


def _parse_vote_response(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the model's JSON response for vote and reason.
    Returns (vote, reason) or (None, None) on failure.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None, None

    vote = data.get("vote")
    reason = data.get("reason")

    if vote not in ("Yes", "No"):
        return None, None
    if not reason or not isinstance(reason, str) or not reason.strip():
        return None, None

    return vote, reason.strip()


def _call_api(
    client, developer_prompt: str, user_prompt: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Make one API call to o4-mini and parse the result.
    Returns (vote, reason) or (None, None) on failure.
    REQ-015, REQ-016, REQ-017, REQ-018.
    """
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="medium",
        response_format={"type": "json_object"},
        messages=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    return _parse_vote_response(content)


def _vote_agent(client, agent: dict, user_prompt: str, question: str) -> dict:
    """
    Call the API for a single agent with one retry on parse failure. REQ-019.
    Returns a response record dict. The record stores the question so cached
    votes are only reused when the same question is asked again.
    """
    developer_prompt = _build_developer_prompt(agent)

    # First attempt
    vote, reason = _call_api(client, developer_prompt, user_prompt)

    # Retry once on parse failure — REQ-019
    if vote is None:
        print(
            f"  [RETRY] Agent {agent['id']} ({agent['name']}): parse failed, retrying...",
            file=sys.stderr,
        )
        try:
            vote, reason = _call_api(client, developer_prompt, user_prompt)
        except Exception as retry_exc:
            print(
                f"  [WARN] Agent {agent['id']} retry failed: {retry_exc}",
                file=sys.stderr,
            )
            vote, reason = None, None

    if vote is None:
        print(
            f"  [WARN] Agent {agent['id']} ({agent['name']}): recording null vote.",
            file=sys.stderr,
        )
        reason = "parse_error"

    return {
        "id": agent["id"],
        "name": agent["name"],
        "vote": vote,
        "reason": reason,
        "question": question,
        "model": MODEL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _load_cached_ids(output_path: Path, question: Optional[str] = None) -> set[int]:
    """
    Read existing responses.jsonl and return the set of agent IDs already processed.
    When ``question`` is given, only records whose stored question matches count as
    cached — a different question invalidates the cache so agents re-vote. REQ-022.
    """
    if not output_path.exists():
        return set()
    cached = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if question is not None and obj.get("question") != question:
                    continue
                cached.add(int(obj["id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return cached


def _append_response(output_path: Path, record: dict) -> None:
    """Append one response record to the JSONL file. REQ-021."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core run_votes function — importable by app.py
# ---------------------------------------------------------------------------

def run_votes(
    agents: list,
    callback: Optional[Callable] = None,
    question: Optional[str] = None,
) -> list:
    """
    Run the voting scenario for all agents concurrently. Skips agents already cached
    for the same question. Uses MAX_CONCURRENCY env var (default 10) to control
    parallel API calls. callback(record) is called after each agent completes (for
    streaming UI). ``question`` overrides the default ballot question when provided.
    Returns list of all response records (cached + new). REQ-014 through REQ-023.
    """
    # REQ-015: Check OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running: export OPENAI_API_KEY=sk-..."
        )

    from openai import OpenAI  # lazy import — only needed at call time
    client = OpenAI(api_key=api_key)

    question = (question or DEFAULT_QUESTION).strip()
    user_prompt = _build_user_prompt(question)

    output_path = DEFAULT_OUTPUT
    cached_ids = _load_cached_ids(output_path, question)

    # Load already-cached responses (for the current question) so we can return a
    # complete list. Records for a different question are ignored here and re-voted.
    existing_records: dict[int, dict] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("question") != question:
                        continue
                    existing_records[int(obj["id"])] = obj
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    total = len(agents)
    results: dict[int, dict] = dict(existing_records)

    # Separate cached vs pending agents
    pending = []
    for agent in agents:
        agent_id = int(agent["id"])
        if agent_id in cached_ids:
            print(f"[cached] {agent['name']} - skipping.")
        else:
            pending.append(agent)

    if not pending:
        print("All agents already cached.")
        ordered = [results[int(a["id"])] for a in agents if int(a["id"]) in results]
        return ordered

    concurrency = MAX_CONCURRENCY
    print(f"Voting {len(pending)} agents with concurrency={concurrency}...")

    # Thread-safe file writes and counter
    write_lock = threading.Lock()
    completed_count = [0]

    def process_agent(agent: dict) -> dict:
        agent_id = int(agent["id"])
        try:
            record = _vote_agent(client, agent, user_prompt, question)
        except Exception as exc:
            print(
                f"  [ERROR] Agent {agent_id} API call failed: {exc}",
                file=sys.stderr,
            )
            record = {
                "id": agent_id,
                "name": agent["name"],
                "vote": None,
                "reason": f"api_error: {exc}",
                "question": question,
                "model": MODEL,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

        with write_lock:
            _append_response(output_path, record)
            results[agent_id] = record
            completed_count[0] += 1
            print(f"[{completed_count[0]}/{len(pending)}] {agent['name']} - {'Yes' if record.get('vote') == 'Yes' else 'No' if record.get('vote') == 'No' else 'error'}")

        if callback is not None:
            callback(record)

        return record

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(process_agent, agent): agent for agent in pending}
        for future in as_completed(futures):
            # Exceptions are captured inside process_agent, but catch any escapes
            try:
                future.result()
            except Exception as exc:
                agent = futures[future]
                print(f"  [FATAL] Agent {agent['id']} thread failed: {exc}", file=sys.stderr)

    # Return in agent order
    ordered = [results[int(a["id"])] for a in agents if int(a["id"]) in results]
    return ordered


# ---------------------------------------------------------------------------
# CLI entry point — REQ-023
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run voting scenario for all agents using OpenAI o4-mini."
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=str(DEFAULT_AGENTS),
        help=f"Path to agents.json (default: {DEFAULT_AGENTS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output path for responses.jsonl (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="The ballot question to ask each agent (default: SF fee cap measure).",
    )
    args = parser.parse_args()

    question = (args.question or DEFAULT_QUESTION).strip()
    user_prompt = _build_user_prompt(question)

    agents_path = Path(args.agents)
    if not agents_path.exists():
        print(
            f"ERROR: Agents file not found: {agents_path}\n"
            "Run generate_population.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(agents_path, encoding="utf-8") as f:
        agents = json.load(f)

    output_path = Path(args.output)

    # REQ-015, REQ-037: Check OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "Export it: export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    cached_ids = _load_cached_ids(output_path, question)
    existing_records: dict[int, dict] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("question") != question:
                        continue
                    existing_records[int(obj["id"])] = obj
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    total = len(agents)
    all_results: dict[int, dict] = dict(existing_records)

    for idx, agent in enumerate(agents, start=1):
        agent_id = int(agent["id"])
        if agent_id in cached_ids:
            print(f"[{idx}/{total}] {agent['name']} - cached, skipping.")
            continue

        # REQ-023: Progress format
        print(f"[{idx}/{total}] {agent['name']} - voting...")
        try:
            record = _vote_agent(client, agent, user_prompt, question)
        except Exception as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            record = {
                "id": agent_id,
                "name": agent["name"],
                "vote": None,
                "reason": f"api_error: {exc}",
                "question": question,
                "model": MODEL,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

        _append_response(output_path, record)
        all_results[agent_id] = record

    print(f"\nDone. Results written to {output_path}")
    null_votes = [r for r in all_results.values() if r.get("vote") is None]
    if null_votes:
        print(
            f"WARNING: {len(null_votes)} agent(s) recorded null votes.",
            file=sys.stderr,
        )
