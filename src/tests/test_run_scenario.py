"""
Unit tests for run_scenario.py — focused on prompt construction.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_scenario import _build_developer_prompt, _describe_experiences


def _make_agent(**kwargs) -> dict:
    agent = {
        "id": 1,
        "name": "Test Person",
        "age": 42,
        "sex": "Female",
        "race_ethnicity": "Asian",
        "neighborhood": "Mission",
        "occupation": "Software Developer",
        "household_income": 150000,
        "tenure": "Renter",
        "ocean": {
            "openness": 7.0,
            "conscientiousness": 6.0,
            "extraversion": 5.0,
            "agreeableness": 6.5,
            "neuroticism": 4.0,
        },
    }
    agent.update(kwargs)
    return agent


class TestDescribeExperiences:
    def test_none_returns_empty(self):
        assert _describe_experiences(None) == ""

    def test_empty_list_returns_empty(self):
        assert _describe_experiences([]) == ""

    def test_renders_bulleted_block(self):
        block = _describe_experiences(["a", "b", "c"])
        assert "Your past experiences with food delivery apps:" in block
        assert "- a" in block and "- b" in block and "- c" in block


class TestDeveloperPromptMemory:
    def test_prompt_omits_block_without_memory(self):
        prompt = _build_developer_prompt(_make_agent())
        assert "past experiences with food delivery apps" not in prompt
        # Personality section and instruction must still be intact.
        assert "Your personality:" in prompt
        assert "When asked about local policy" in prompt

    def test_prompt_includes_experiences_when_present(self):
        agent = _make_agent(delivery_experiences=[
            "I tipped a rain-soaked courier extra one December night.",
            "A restaurant I love dropped off the app over high fees.",
            "My order was marked delivered but never arrived.",
        ])
        prompt = _build_developer_prompt(agent)
        assert "Your past experiences with food delivery apps:" in prompt
        assert "rain-soaked courier" in prompt
        assert "never arrived" in prompt
        # Ordering: experiences appear after personality, before the instruction.
        assert prompt.index("Your personality:") < prompt.index("past experiences")
        assert prompt.index("past experiences") < prompt.index("When asked about local policy")

    def test_empty_experiences_list_leaves_prompt_unchanged(self):
        base = _build_developer_prompt(_make_agent())
        with_empty = _build_developer_prompt(_make_agent(delivery_experiences=[]))
        assert base == with_empty
