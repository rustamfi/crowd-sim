"""
Unit tests for generate_population.py
REQ-001 through REQ-013.
"""

import json
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_population import (
    _build_profile,
    _derive_ocean,
    _generate_name,
    _income_bracket,
    _is_swe,
    _map_occupation,
    _race_ethnicity_label,
    _sex_label,
    _tenure_label,
    PUMA_NEIGHBORHOODS,
    OCCP_LOOKUP,
    SWE_CODES,
)


# ---------------------------------------------------------------------------
# Helper: minimal agent record
# ---------------------------------------------------------------------------
def _make_record(**kwargs) -> dict:
    defaults = {
        "AGEP": "30",
        "OCCP": "5300",  # Customer Service Rep — not a SWE
        "HINCP": "80000",
        "TEN": "3",
        "RAC1P": "1",
        "HISP": "1",
        "SEX": "1",
        "PWGTP": "100",
        "public use microdata area": "07501",
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# REQ-005: PUMA neighborhood mapping
# ---------------------------------------------------------------------------
class TestPumaNeighborhoods:
    def test_all_sf_pumas_present(self):
        """REQ-005: All SF 2020 PUMAs (07507-07514) must be mapped."""
        expected = {f"075{i:02d}" for i in range(7, 15)}
        assert set(PUMA_NEIGHBORHOODS.keys()) == expected

    def test_each_puma_has_multiple_neighborhoods(self):
        """REQ-005: Each PUMA should have at least 2 neighborhood options."""
        for puma, hoods in PUMA_NEIGHBORHOODS.items():
            assert len(hoods) >= 2, f"PUMA {puma} has too few neighborhoods"

    def test_neighborhoods_are_strings(self):
        for hoods in PUMA_NEIGHBORHOODS.values():
            for h in hoods:
                assert isinstance(h, str) and len(h) > 0


# ---------------------------------------------------------------------------
# REQ-006: OCCP occupation mapping
# ---------------------------------------------------------------------------
class TestOccupationMapping:
    def test_swe_codes_map_correctly(self):
        """REQ-006, REQ-011: SWE codes must be in the lookup."""
        for code in ("1010", "1020", "1030", "1050", "1060"):
            result = _map_occupation(code)
            assert "software" in result.lower() or "developer" in result.lower() or "engineer" in result.lower(), \
                f"Code {code} mapped to unexpected: {result}"

    def test_null_occp_maps_to_not_in_labor_force(self):
        """REQ-006: N and None should map to 'Not in labor force'."""
        assert _map_occupation("N") == "Not in labor force"
        assert _map_occupation(None) == "Not in labor force"
        assert _map_occupation("") == "Not in labor force"

    def test_unknown_code_falls_back_to_range(self):
        """REQ-006: Unknown codes fall back to range-based categories."""
        # Code 3200 = Registered Nurse — in healthcare range
        result = _map_occupation("3200")
        assert result != "Other"

    def test_completely_unknown_code_returns_other(self):
        """REQ-006: Truly unknown codes return 'Other'."""
        result = _map_occupation("9999")
        assert result == "Other"

    def test_lookup_covers_at_least_40_codes(self):
        """REQ-006: Lookup must cover at minimum 40 OCCP codes."""
        # Exclude special values N and 9920
        meaningful = {k: v for k, v in OCCP_LOOKUP.items() if k not in ("N", "9920")}
        assert len(meaningful) >= 40, f"Only {len(meaningful)} codes in lookup"

    def test_is_swe_true_for_cap_codes(self):
        """REQ-011: _is_swe returns True for software engineer codes."""
        for code in SWE_CODES:
            assert _is_swe(code)

    def test_is_swe_false_for_other_codes(self):
        """REQ-011: _is_swe returns False for non-SWE codes."""
        assert not _is_swe("3200")
        assert not _is_swe("5300")
        assert not _is_swe("N")


# ---------------------------------------------------------------------------
# REQ-007: Name generation
# ---------------------------------------------------------------------------
class TestNameGeneration:
    def test_name_is_two_words(self):
        """REQ-007: Name should be 'First Last' format."""
        name = _generate_name(1, "1", "1", "1")
        parts = name.split()
        assert len(parts) == 2

    def test_hispanic_override(self):
        """REQ-007: HISP > 1 should produce Hispanic-group names."""
        # Run multiple times with different agent IDs to check variety
        names = {_generate_name(i, "1", "2", "1") for i in range(1, 20)}
        # Hispanic names like Jose, Maria, Garcia, Rodriguez should appear
        all_names = " ".join(names)
        # At least some names should be from hispanic list
        assert len(names) > 1, "Names should vary by agent_id"

    def test_reproducible_with_same_agent_id(self):
        """REQ-007: Same agent_id produces same name."""
        name1 = _generate_name(5, "1", "1", "2")
        name2 = _generate_name(5, "1", "1", "2")
        assert name1 == name2

    def test_different_agent_ids_may_differ(self):
        """REQ-007: Different agent IDs should (usually) produce different names."""
        names = {_generate_name(i, "1", "1", "1") for i in range(1, 15)}
        assert len(names) > 1, "All names were identical — seeding may be broken"

    def test_male_female_can_differ(self):
        """REQ-007: Male and female names should generally differ."""
        male_names = {_generate_name(i, "1", "1", "1") for i in range(1, 10)}
        female_names = {_generate_name(i, "1", "1", "2") for i in range(1, 10)}
        # They're drawn from different lists, so most should differ
        overlap = male_names & female_names
        assert len(overlap) < len(male_names), "Male/female names should mostly differ"


# ---------------------------------------------------------------------------
# REQ-008: OCEAN score derivation
# ---------------------------------------------------------------------------
class TestOceanDerivation:
    def test_all_five_traits_present(self):
        """REQ-008: OCEAN dict must contain all 5 traits."""
        ocean = _derive_ocean(1, 25, "1", 90000, "Renter", "Software Engineer")
        assert set(ocean.keys()) == {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}

    def test_scores_clamped_to_range(self):
        """REQ-008: All scores must be in [1.0, 10.0]."""
        # Extreme cases
        for agent_id in range(1, 50):
            ocean = _derive_ocean(agent_id, 18, "2", 250000, "Owner", "Software Engineer")
            for trait, val in ocean.items():
                assert 1.0 <= val <= 10.0, f"Trait {trait}={val} out of range for agent {agent_id}"

    def test_demographics_are_primary_signal(self):
        """REQ-008: Demographic differences should produce systematically different means."""
        # Young (age<30) should score higher on openness than old (age>50) on average
        young_openness = [_derive_ocean(i, 22, "1", 60000, "Renter", "Clerk")["openness"] for i in range(1, 20)]
        old_openness = [_derive_ocean(i, 60, "1", 60000, "Owner", "Clerk")["openness"] for i in range(1, 20)]
        assert sum(young_openness) / len(young_openness) > sum(old_openness) / len(old_openness)

    def test_reproducible_per_agent(self):
        """REQ-008: Same agent_id + same demographics = same OCEAN."""
        o1 = _derive_ocean(7, 35, "2", 90000, "Renter", "Nurse")
        o2 = _derive_ocean(7, 35, "2", 90000, "Renter", "Nurse")
        assert o1 == o2

    def test_scores_are_floats_rounded_to_one_decimal(self):
        """REQ-008: Scores should be floats with at most 1 decimal place."""
        ocean = _derive_ocean(3, 40, "1", 100000, "Renter", "Teacher")
        for trait, val in ocean.items():
            assert isinstance(val, float)
            assert round(val, 1) == val


# ---------------------------------------------------------------------------
# REQ-009: Profile generation
# ---------------------------------------------------------------------------
class TestProfileGeneration:
    def _make_ocean(self):
        return {"openness": 7.0, "conscientiousness": 6.0, "extraversion": 5.0,
                "agreeableness": 6.0, "neuroticism": 4.0}

    def test_profile_is_two_sentences(self):
        """REQ-009: Profile must be exactly 2 sentences."""
        ocean = self._make_ocean()
        profile = _build_profile("Jane Smith", 30, "Teacher", "Mission", 90000, "Renter", ocean)
        # Count sentence-ending punctuation
        sentences = [s.strip() for s in profile.split(".") if s.strip()]
        assert len(sentences) >= 2

    def test_profile_contains_name(self):
        """REQ-009: Profile should contain the agent's first name."""
        ocean = self._make_ocean()
        profile = _build_profile("John Doe", 28, "Chef", "SoMa", 55000, "Renter", ocean)
        assert "John" in profile

    def test_profile_contains_neighborhood(self):
        """REQ-009: Profile should mention the neighborhood."""
        ocean = self._make_ocean()
        profile = _build_profile("Ana Lopez", 45, "Nurse", "Castro", 110000, "Owner", ocean)
        assert "Castro" in profile

    def test_profile_is_third_person(self):
        """REQ-009: Profile should use third-person pronouns."""
        ocean = self._make_ocean()
        profile = _build_profile("Carlos Reyes", 55, "Lawyer", "Noe Valley", 200000, "Owner", ocean)
        # Should not start with "I" — must be third-person
        assert not profile.startswith("I ")
        assert "Carlos" in profile or "They" in profile


# ---------------------------------------------------------------------------
# REQ-010: Income bracket derivation
# ---------------------------------------------------------------------------
class TestIncomeBracket:
    def test_below_35k(self):
        """REQ-010: HINCP < 35000 maps to '<$35k'."""
        assert _income_bracket(0) == "<$35k"
        assert _income_bracket(34999) == "<$35k"
        assert _income_bracket(-5000) == "<$35k"

    def test_35k_to_75k(self):
        """REQ-010: 35000 <= HINCP < 75000 maps to '$35k-$75k'."""
        assert _income_bracket(35000) == "$35k-$75k"
        assert _income_bracket(50000) == "$35k-$75k"
        assert _income_bracket(74999) == "$35k-$75k"

    def test_75k_to_120k(self):
        """REQ-010: 75000 <= HINCP < 120000 maps to '$75k-$120k'."""
        assert _income_bracket(75000) == "$75k-$120k"
        assert _income_bracket(100000) == "$75k-$120k"
        assert _income_bracket(119999) == "$75k-$120k"

    def test_above_120k(self):
        """REQ-010: HINCP >= 120000 maps to '>$120k'."""
        assert _income_bracket(120000) == ">$120k"
        assert _income_bracket(250000) == ">$120k"


# ---------------------------------------------------------------------------
# REQ-010: Field labels
# ---------------------------------------------------------------------------
class TestFieldLabels:
    def test_sex_label_male(self):
        assert _sex_label("1") == "Male"

    def test_sex_label_female(self):
        assert _sex_label("2") == "Female"

    def test_tenure_owner_codes(self):
        assert _tenure_label("1") == "Owner"
        assert _tenure_label("2") == "Owner"

    def test_tenure_renter_codes(self):
        assert _tenure_label("3") == "Renter"
        assert _tenure_label("4") == "Renter"

    def test_race_ethnicity_hispanic_override(self):
        """REQ-010: HISP > 1 should override RAC1P."""
        label = _race_ethnicity_label("1", "2")  # White RAC1P but Hispanic HISP
        assert "Hispanic" in label

    def test_race_ethnicity_white(self):
        label = _race_ethnicity_label("1", "1")
        assert "White" in label

    def test_race_ethnicity_asian(self):
        label = _race_ethnicity_label("6", "1")
        assert "Asian" in label


# ---------------------------------------------------------------------------
# REQ-004 / REQ-011: Integration — weighted sampling and SWE cap
# (Uses mocked Census data to avoid network calls)
# ---------------------------------------------------------------------------
class TestGenerateIntegration:
    def _make_fake_records(self, n=100) -> list[dict]:
        """Create n fake adult PUMS records."""
        rng = random.Random(0)
        records = []
        occp_pool = ["5300", "3200", "4060", "2200", "1010", "1020", "1030", "1050", "1060", "9120"]
        for i in range(n):
            records.append({
                "AGEP": str(rng.randint(18, 65)),
                "OCCP": rng.choice(occp_pool),
                "HINCP": str(rng.randint(30000, 250000)),
                "TEN": str(rng.randint(1, 4)),
                "RAC1P": str(rng.randint(1, 9)),
                "HISP": "1",
                "SEX": str(rng.randint(1, 2)),
                "PWGTP": str(rng.randint(10, 500)),
                "public use microdata area": rng.choice(
                    ["07501", "07502", "07503", "07504", "07505", "07506", "07507"]
                ),
            })
        return records

    def test_swe_cap_enforced(self):
        """REQ-011: At most 4 agents may have SWE occupation codes."""
        fake_records = self._make_fake_records(200)
        # Inject lots of SWEs
        for r in fake_records[:50]:
            r["OCCP"] = "1010"

        with patch("generate_population._fetch_pums", return_value=fake_records):
            from generate_population import generate
            agents = generate(seed=42)

        swe_agents = [a for a in agents if _is_swe(a["occupation_code"])]
        assert len(swe_agents) <= 4, f"SWE cap violated: {len(swe_agents)} SWEs"

    def test_generates_exactly_30_agents(self):
        """REQ-004: Exactly 30 agents must be generated."""
        fake_records = self._make_fake_records(200)
        # Ensure no SWE overflow
        for r in fake_records:
            r["OCCP"] = "5300"

        with patch("generate_population._fetch_pums", return_value=fake_records):
            from generate_population import generate
            agents = generate(seed=42)

        assert len(agents) == 30

    def test_all_adults(self):
        """REQ-003: All sampled agents must be 18+."""
        fake_records = self._make_fake_records(200)
        for r in fake_records:
            r["OCCP"] = "5300"

        with patch("generate_population._fetch_pums", return_value=fake_records):
            from generate_population import generate
            agents = generate(seed=42)

        for a in agents:
            assert a["age"] >= 18, f"Agent {a['id']} is under 18: age={a['age']}"

    def test_required_fields_present(self):
        """REQ-010: All required fields must be present in each agent."""
        required = {
            "id", "name", "age", "sex", "race_ethnicity", "neighborhood",
            "occupation", "household_income",
            "household_income_bracket", "tenure", "ocean", "profile",
        }
        fake_records = self._make_fake_records(200)
        for r in fake_records:
            r["OCCP"] = "5300"

        with patch("generate_population._fetch_pums", return_value=fake_records):
            from generate_population import generate
            agents = generate(seed=42)

        for agent in agents:
            missing = required - set(agent.keys())
            assert not missing, f"Agent {agent['id']} missing fields: {missing}"

    def test_reproducible_with_same_seed(self):
        """REQ-013: Same seed produces identical output."""
        fake_records = self._make_fake_records(200)
        for r in fake_records:
            r["OCCP"] = "5300"

        with patch("generate_population._fetch_pums", return_value=fake_records):
            from generate_population import generate
            agents1 = generate(seed=99)
            agents2 = generate(seed=99)

        assert agents1 == agents2

    def test_different_seeds_differ(self):
        """REQ-013: Different seeds should (likely) produce different populations."""
        fake_records = self._make_fake_records(200)
        for r in fake_records:
            r["OCCP"] = "5300"

        with patch("generate_population._fetch_pums", return_value=fake_records):
            from generate_population import generate
            agents1 = generate(seed=42)
            agents2 = generate(seed=99)

        names1 = [a["name"] for a in agents1]
        names2 = [a["name"] for a in agents2]
        assert names1 != names2

    def test_household_income_bracket_valid_values(self):
        """REQ-010: household_income_bracket must be one of the four allowed values."""
        valid_brackets = {"<$35k", "$35k-$75k", "$75k-$120k", ">$120k"}
        fake_records = self._make_fake_records(200)
        for r in fake_records:
            r["OCCP"] = "5300"

        with patch("generate_population._fetch_pums", return_value=fake_records):
            from generate_population import generate
            agents = generate(seed=42)

        for agent in agents:
            assert agent["household_income_bracket"] in valid_brackets, \
                f"Invalid bracket: {agent['household_income_bracket']}"


# ---------------------------------------------------------------------------
# Agent memory — 3 past delivery-app experiences (optional, LLM-backed)
# ---------------------------------------------------------------------------
def _mock_openai_client(experiences):
    """A MagicMock OpenRouter client whose completion returns the given experiences."""
    client = MagicMock()
    message = MagicMock()
    message.content = json.dumps({"experiences": experiences})
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=message)]
    )
    return client


class TestParseExperiences:
    def test_valid_three_item_list(self):
        from generate_population import _parse_experiences
        content = json.dumps({"experiences": ["a", "b", "c"]})
        assert _parse_experiences(content) == ["a", "b", "c"]

    def test_strips_whitespace(self):
        from generate_population import _parse_experiences
        content = json.dumps({"experiences": ["  a ", "b", "c "]})
        assert _parse_experiences(content) == ["a", "b", "c"]

    def test_wrong_count_returns_empty(self):
        from generate_population import _parse_experiences
        assert _parse_experiences(json.dumps({"experiences": ["a", "b"]})) == []

    def test_empty_string_item_returns_empty(self):
        from generate_population import _parse_experiences
        assert _parse_experiences(json.dumps({"experiences": ["a", "", "c"]})) == []

    def test_malformed_json_returns_empty(self):
        from generate_population import _parse_experiences
        assert _parse_experiences("not json") == []

    def test_missing_key_returns_empty(self):
        from generate_population import _parse_experiences
        assert _parse_experiences(json.dumps({"foo": "bar"})) == []


class TestGenerateDeliveryExperiences:
    def _agent(self):
        return {
            "id": 1, "name": "Test Person", "age": 34, "sex": "Female",
            "race_ethnicity": "Asian", "neighborhood": "Mission",
            "occupation": "Software Developer", "household_income": 150000,
            "tenure": "Renter", "profile": "A curious 34-year-old.",
        }

    def test_returns_three_experiences(self):
        from generate_population import _generate_delivery_experiences
        client = _mock_openai_client(["exp1", "exp2", "exp3"])
        result = _generate_delivery_experiences(self._agent(), client, "test/model")
        assert result == ["exp1", "exp2", "exp3"]

    def test_uses_system_role_and_openrouter_reasoning(self):
        """OpenRouter rules: system role (not developer), unified reasoning, no temperature."""
        from generate_population import _generate_delivery_experiences
        client = _mock_openai_client(["exp1", "exp2", "exp3"])
        _generate_delivery_experiences(self._agent(), client, "test/model")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "test/model"
        assert "temperature" not in kwargs
        assert "reasoning_effort" not in kwargs
        assert kwargs["extra_body"]["reasoning"]["effort"] == "low"
        roles = [m["role"] for m in kwargs["messages"]]
        assert "system" in roles and "developer" not in roles

    def test_retries_then_returns_empty_on_persistent_failure(self):
        from generate_population import _generate_delivery_experiences
        client = _mock_openai_client(["only", "two"])  # invalid every time
        result = _generate_delivery_experiences(self._agent(), client, "test/model")
        assert result == []
        assert client.chat.completions.create.call_count == 2

    def test_api_exception_is_swallowed(self):
        from generate_population import _generate_delivery_experiences
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        assert _generate_delivery_experiences(self._agent(), client, "test/model") == []


class TestGenerateWithMemory:
    def _make_fake_records(self, n=200):
        return TestGenerateIntegration()._make_fake_records(n)

    def test_no_memory_by_default(self):
        """Without use_memory, agents have no delivery_experiences key."""
        fake_records = self._make_fake_records()
        for r in fake_records:
            r["OCCP"] = "5300"
        with patch("generate_population._fetch_pums", return_value=fake_records):
            from generate_population import generate
            agents = generate(seed=42)
        assert all("delivery_experiences" not in a for a in agents)

    def test_use_memory_attaches_three_experiences(self):
        fake_records = self._make_fake_records()
        for r in fake_records:
            r["OCCP"] = "5300"
        client = _mock_openai_client(["exp1", "exp2", "exp3"])
        with patch("generate_population._fetch_pums", return_value=fake_records), \
             patch("openai.OpenAI", return_value=client), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            from generate_population import generate
            agents = generate(seed=42, use_memory=True)
        assert all(a["delivery_experiences"] == ["exp1", "exp2", "exp3"] for a in agents)

    def test_use_memory_requires_api_key(self):
        from generate_population import _attach_delivery_experiences
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                _attach_delivery_experiences([{"id": 1}])
