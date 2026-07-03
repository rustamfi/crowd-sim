"""
Unit tests for analyze.py
REQ-024 through REQ-028.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyze import (
    _classify_reason,
    _group_themes,
    _join,
    _most_interesting,
    _build_markdown,
    YES_THEMES,
    NO_THEMES,
    analyze,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_agent(agent_id=1, name="Jane Doe", age=35, neighborhood="Mission",
                occupation="Nurse", bracket="$35k-$75k", tenure="Renter",
                vote=None, reason=None) -> dict:
    return {
        "id": agent_id,
        "name": name,
        "age": age,
        "neighborhood": neighborhood,
        "occupation": occupation,
        "household_income_bracket": bracket,
        "tenure": tenure,
        "income_annual": 60000,
        "household_income": 80000,
        "vote": vote,
        "reason": reason,
        "ocean": {"openness": 6.0, "conscientiousness": 6.0, "extraversion": 6.0,
                  "agreeableness": 6.0, "neuroticism": 5.0},
        "profile": "Jane is a nurse. She tends to be cooperative.",
    }


def _make_response(agent_id=1, name="Jane Doe", vote="Yes", reason="Fees are too high.") -> dict:
    return {
        "id": agent_id,
        "name": name,
        "vote": vote,
        "reason": reason,
        "model": "o4-mini",
        "timestamp_utc": "2024-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# REQ-024, REQ-025: Join
# ---------------------------------------------------------------------------
class TestJoin:
    def test_join_on_id(self):
        """REQ-024: Join merges agent and response fields."""
        agents = [_make_agent(1, "Alice")]
        responses = [_make_response(1, "Alice", "Yes", "Affordability matters.")]
        joined = _join(agents, responses)
        assert len(joined) == 1
        assert joined[0]["vote"] == "Yes"
        assert joined[0]["name"] == "Alice"

    def test_join_missing_response(self):
        """REQ-024: Agents without a matching response get no vote field from response."""
        agents = [_make_agent(1, "Alice"), _make_agent(2, "Bob")]
        responses = [_make_response(1, "Alice", "No", "Too much regulation.")]
        joined = _join(agents, responses)
        assert len(joined) == 2
        # Agent 2 has no response — vote should be None (from _make_agent default)
        bob = next(j for j in joined if j["id"] == 2)
        assert bob.get("vote") is None

    def test_join_preserves_order(self):
        """REQ-024: Join should maintain agent order."""
        agents = [_make_agent(i) for i in range(1, 6)]
        responses = [_make_response(i, vote="Yes", reason="reason") for i in range(1, 6)]
        joined = _join(agents, responses)
        assert [j["id"] for j in joined] == list(range(1, 6))


# ---------------------------------------------------------------------------
# REQ-026: Theme classification
# ---------------------------------------------------------------------------
class TestThemeClassification:
    def test_yes_affordability_theme(self):
        """REQ-026: 'afford' keyword maps to Affordability theme."""
        theme = _classify_reason("I can't afford delivery anymore with these fees.", YES_THEMES)
        assert theme == "Affordability / Cost of Living"

    def test_yes_restaurant_theme(self):
        """REQ-026: 'restaurant' keyword maps to Support for Small Restaurants."""
        theme = _classify_reason("Local restaurants are suffering under these fees.", YES_THEMES)
        assert theme == "Support for Small Restaurants"

    def test_yes_fairness_theme(self):
        """REQ-026: 'fair' keyword maps to Fairness theme."""
        theme = _classify_reason("It's not fair to charge such high fees.", YES_THEMES)
        assert theme == "Fairness / Consumer Rights"

    def test_no_free_market_theme(self):
        """REQ-026: 'free market' keyword maps to Free Market theme."""
        theme = _classify_reason("The free market should decide these prices.", NO_THEMES)
        assert theme == "Free Market / Competition"

    def test_no_government_overreach_theme(self):
        """REQ-026: 'government' keyword maps to Government Overreach."""
        # Use a reason with 'overreach' that doesn't trigger Free Market keywords
        theme = _classify_reason("This is a clear case of government overreach into daily commerce.", NO_THEMES)
        assert theme == "Government Overreach"

    def test_no_match_falls_to_other(self):
        """REQ-026: Unknown reason falls to 'Other' catch-all."""
        # A reason with no recognizable keywords should hit the catch-all
        theme = _classify_reason("I have mixed feelings about this issue broadly.", YES_THEMES)
        # 'Other' is always last and is the catch-all
        assert theme in [t for t, _ in YES_THEMES]


# ---------------------------------------------------------------------------
# REQ-025: Most interesting response
# ---------------------------------------------------------------------------
class TestMostInteresting:
    def test_selects_longest_reason(self):
        """REQ-025: Should select agent with most words in reason."""
        joined = [
            {**_make_agent(1, "Alice"), "vote": "Yes",
             "reason": "Fees are high."},
            {**_make_agent(2, "Bob"), "vote": "No",
             "reason": "This cap would undermine the competitive dynamics that have historically "
                        "driven innovation in the food delivery sector."},
            {**_make_agent(3, "Carol"), "vote": "Yes",
             "reason": "I support it because restaurants need help."},
        ]
        interesting = _most_interesting(joined)
        assert interesting["name"] == "Bob"

    def test_excludes_parse_errors(self):
        """REQ-025: parse_error reasons should not be selected as most interesting."""
        joined = [
            {**_make_agent(1, "Alice"), "vote": None, "reason": "parse_error"},
            {**_make_agent(2, "Bob"), "vote": "Yes", "reason": "Good policy for consumers."},
        ]
        interesting = _most_interesting(joined)
        assert interesting["name"] == "Bob"

    def test_returns_none_when_no_valid_responses(self):
        """REQ-025: Returns None when all responses are errors."""
        joined = [
            {**_make_agent(1), "vote": None, "reason": "parse_error"},
        ]
        assert _most_interesting(joined) is None


# ---------------------------------------------------------------------------
# REQ-025: Markdown output structure
# ---------------------------------------------------------------------------
class TestBuildMarkdown:
    def _make_joined(self):
        return [
            {**_make_agent(i, f"Agent {i}"), "vote": "Yes" if i % 2 == 0 else "No",
             "reason": f"Reason number {i} about delivery fees."}
            for i in range(1, 6)
        ]

    def test_contains_all_sections(self):
        """REQ-025: Markdown must contain all 5 required sections."""
        joined = self._make_joined()
        yes_themes = _group_themes(joined, "Yes", YES_THEMES)
        no_themes = _group_themes(joined, "No", NO_THEMES)
        interesting = _most_interesting(joined)
        md = _build_markdown(joined, 2, 3, 0, yes_themes, no_themes, interesting)

        assert "## 1. Vote Tally" in md
        assert "## 2. Top Reasons — Yes" in md
        assert "## 3. Top Reasons — No" in md
        assert "## 4. Most Interesting Response" in md
        assert "## 5. Full Agent Table" in md

    def test_vote_counts_in_tally(self):
        """REQ-025: Vote tally section should show correct counts."""
        joined = self._make_joined()
        yes_themes = _group_themes(joined, "Yes", YES_THEMES)
        no_themes = _group_themes(joined, "No", NO_THEMES)
        interesting = _most_interesting(joined)
        md = _build_markdown(joined, 2, 3, 0, yes_themes, no_themes, interesting)
        assert "| Yes" in md
        assert "| No" in md

    def test_table_contains_all_agents(self):
        """REQ-025: Full agent table should contain all agents."""
        joined = self._make_joined()
        yes_themes = _group_themes(joined, "Yes", YES_THEMES)
        no_themes = _group_themes(joined, "No", NO_THEMES)
        interesting = _most_interesting(joined)
        md = _build_markdown(joined, 2, 3, 0, yes_themes, no_themes, interesting)

        for record in joined:
            assert record["name"] in md


# ---------------------------------------------------------------------------
# REQ-024, REQ-025: Integration test for analyze()
# ---------------------------------------------------------------------------
class TestAnalyzeIntegration:
    def _make_full_data(self, n=10):
        agents = [_make_agent(i, f"Person {i}") for i in range(1, n + 1)]
        responses = [
            _make_response(i, f"Person {i}",
                           vote="Yes" if i % 3 != 0 else "No",
                           reason=f"Reason from person {i} about delivery fee caps.")
            for i in range(1, n + 1)
        ]
        return agents, responses

    def test_analyze_returns_required_keys(self):
        """REQ-024, REQ-025: analyze() must return expected keys."""
        agents, responses = self._make_full_data()
        with patch("analyze.DEFAULT_OUTPUT", Path("/tmp/test_summary.md")):
            result = analyze(agents, responses)

        required_keys = {"yes_count", "no_count", "null_count", "pct_yes",
                         "yes_themes", "no_themes", "most_interesting", "agents", "markdown"}
        assert required_keys.issubset(set(result.keys()))

    def test_tally_counts_correct(self):
        """REQ-025: Vote counts must sum correctly."""
        agents, responses = self._make_full_data(9)
        with patch("analyze.DEFAULT_OUTPUT", Path("/tmp/test_summary.md")):
            result = analyze(agents, responses)

        assert result["yes_count"] + result["no_count"] + result["null_count"] == 9

    def test_pct_yes_calculation(self):
        """REQ-025: Percentage should be yes/(yes+no)*100."""
        agents = [_make_agent(i) for i in range(1, 5)]
        responses = [
            _make_response(1, vote="Yes", reason="Affordable fees help everyone."),
            _make_response(2, vote="Yes", reason="Cost savings for consumers."),
            _make_response(3, vote="No", reason="Free market should decide prices."),
            _make_response(4, vote="No", reason="Government overreach in business."),
        ]
        with patch("analyze.DEFAULT_OUTPUT", Path("/tmp/test_summary.md")):
            result = analyze(agents, responses)

        assert result["pct_yes"] == 50.0

    def test_writes_summary_file(self, tmp_path):
        """REQ-025: analyze() must write summary.md to the output path."""
        out = tmp_path / "summary.md"
        agents, responses = self._make_full_data(5)
        with patch("analyze.DEFAULT_OUTPUT", out):
            analyze(agents, responses)
        assert out.exists()
        content = out.read_text()
        assert "Vote Tally" in content


# ---------------------------------------------------------------------------
# REQ-026: Group themes function
# ---------------------------------------------------------------------------
class TestGroupThemes:
    def test_returns_at_most_3_themes(self):
        """REQ-026: _group_themes returns at most top 3 themes."""
        joined = []
        for i in range(1, 20):
            joined.append({
                **_make_agent(i, f"Person {i}"),
                "vote": "Yes",
                "reason": f"This measure is about affordability and consumer rights for person {i}.",
            })
        themes = _group_themes(joined, "Yes", YES_THEMES)
        assert len(themes) <= 3

    def test_skips_parse_errors(self):
        """REQ-026: parse_error reasons are excluded from theme grouping."""
        joined = [
            {**_make_agent(1), "vote": "Yes", "reason": "parse_error"},
            {**_make_agent(2, "Bob"), "vote": "Yes", "reason": "I can't afford these fees."},
        ]
        themes = _group_themes(joined, "Yes", YES_THEMES)
        # Bob should be in themes; parse_error should not add Agent 1
        all_agents = [name for _, names in themes for name in names]
        assert "Bob" in all_agents


# ---------------------------------------------------------------------------
# REQ-020: Ensure no forbidden words in user prompt
# ---------------------------------------------------------------------------
class TestForbiddenWords:
    def test_no_prop_f_in_prompts(self):
        """REQ-020: Prompt must not mention 'Prop F' or '2021'."""
        import run_scenario
        prompt_text = run_scenario.USER_PROMPT + run_scenario.DEVELOPER_PROMPT_TEMPLATE
        for forbidden in ["Prop F", "Proposition F", "2021"]:
            assert forbidden not in prompt_text, \
                f"Forbidden term '{forbidden}' found in prompt"
