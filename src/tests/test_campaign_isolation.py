"""
Tests for the simulation-campaign isolation guarantee.

A campaign votes each seed against its own file via ``run_votes(..., output_path=)``.
These tests assert that passing ``output_path`` (a) writes there and (b) never
touches the shared ``DEFAULT_OUTPUT`` (results/responses.jsonl) — the property the
campaign relies on to keep the single-run state intact and to avoid cross-seed
vote-cache collisions.
"""

import json
import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import run_scenario


def _make_agents(n: int = 3) -> list[dict]:
    return [{"id": i, "name": f"Agent {i}"} for i in range(1, n + 1)]


def _patch_llm(monkeypatch):
    """Stub out the OpenRouter client and per-agent vote so no network call happens."""
    monkeypatch.setattr(run_scenario.llm, "require_client", lambda: object())
    monkeypatch.setattr(run_scenario.llm, "resolve_model", lambda m: m or "test/model")

    def fake_vote(client, agent, user_prompt, question, model):
        return {
            "id": int(agent["id"]),
            "name": agent["name"],
            "vote": "Yes" if int(agent["id"]) % 2 else "No",
            "reason": "stubbed",
            "question": question,
            "model": model,
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(run_scenario, "_vote_agent", fake_vote)


def test_output_path_writes_isolated_file_and_leaves_default_untouched(tmp_path, monkeypatch):
    _patch_llm(monkeypatch)

    # Redirect DEFAULT_OUTPUT to a sentinel that must NOT be written.
    default_sentinel = tmp_path / "responses.jsonl"
    monkeypatch.setattr(run_scenario, "DEFAULT_OUTPUT", default_sentinel)

    seed_file = tmp_path / "campaign_seed.jsonl"
    agents = _make_agents(3)

    results = run_scenario.run_votes(agents, question="Q?", model="m", output_path=seed_file)

    # All agents voted, results returned in agent order.
    assert [r["id"] for r in results] == [1, 2, 3]

    # Votes landed in the isolated file...
    assert seed_file.exists()
    lines = [json.loads(l) for l in seed_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    # ...and the shared default file was never created.
    assert not default_sentinel.exists()


def test_separate_seed_files_do_not_share_vote_cache(tmp_path, monkeypatch):
    """Two seeds with the same agent ids must both fully vote (no cross-seed cache hit)."""
    _patch_llm(monkeypatch)
    monkeypatch.setattr(run_scenario, "DEFAULT_OUTPUT", tmp_path / "responses.jsonl")

    agents = _make_agents(3)
    file_a = tmp_path / "seed_a.jsonl"
    file_b = tmp_path / "seed_b.jsonl"

    run_scenario.run_votes(agents, question="Q?", model="m", output_path=file_a)
    res_b = run_scenario.run_votes(agents, question="Q?", model="m", output_path=file_b)

    # Seed B voted all 3 despite identical ids/question/model in seed A's file.
    assert len(res_b) == 3
    assert len([l for l in file_b.read_text().splitlines() if l.strip()]) == 3
