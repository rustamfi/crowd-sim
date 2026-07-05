"""
SF Crowd Voting Simulator — Results Analyzer
=============================================
Joins agents.json with responses.jsonl on id, computes vote tallies,
groups yes/no reasons by keyword-based themes, identifies the most
interesting response, and writes results/summary.md.

Most Interesting Response (REQ-025)
-----------------------------------
Preferred: when an OpenAI client is supplied to analyze() and both sides voted,
the reasoning model picks the response most likely to change an opposing voter's mind — the
argument most persuasive "across the aisle" — and authors a one-sentence reason
for the pick. If no client is given, everyone voted the same way, or the LLM call
fails, we fall back to a deterministic heuristic: the response with the longest
reason by word count. Either way the chosen record carries a ``selection_reason``.

REQ-024 through REQ-028.

Theme Grouping Logic (REQ-026)
-------------------------------
Preferred: when an OpenAI client is supplied to analyze(), the actual Yes/No
reasons are clustered into up to 3 themes by the reasoning model, so the labels fit whatever
ballot question was asked. If no client is given, or the LLM call fails, we fall
back to the keyword-based buckets below (tuned for the delivery-fee scenario).

Keyword fallback: themes are detected via keyword/substring matching on the
lowercased reason text. Themes are checked in priority order; a reason matches
the FIRST theme whose keywords appear. If no theme matches, it falls into "Other".

YES themes (in priority order):
  1. Affordability / Cost of Living
     keywords: afford, cost, expensive, price, low income, budget, cheap, saving, wage, worker
  2. Support for Small Restaurants
     keywords: restaurant, local, small business, neighborhood eatery, owner, establishment
  3. Fairness / Consumer Rights
     keywords: fair, consumer, right, protect, exploitation, gouging, excessive fee, cap, limit
  4. Access / Equity
     keywords: access, equity, equit, underserved, community, delivery worker, gig worker
  5. Economic Pressure
     keywords: economic, financial, pressure, struggle, hardship, survive, margin
  6. Other (catch-all)

NO themes (in priority order):
  1. Free Market / Competition
     keywords: free market, competition, competi, market force, natural, innovation, private
  2. Business Viability / App Quality
     keywords: app, service, quality, viable, sustain, revenue, platform, operate, feature
  3. Job Losses / Driver Earnings
     keywords: job, driver, earning, gig, lay off, layoff, employment, worker pay, reduce pay
  4. Government Overreach
     keywords: government, overreach, interfere, interference, regulation, over-regulate, mandate
  5. Unintended Consequences
     keywords: unintend, consequence, backfire, result, effect, higher price, less selection
  6. Other (catch-all)
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import llm  # noqa: E402  — shared OpenRouter client + model registry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DEFAULT_AGENTS = RESULTS_DIR / "agents.json"
DEFAULT_RESPONSES = RESULTS_DIR / "responses.jsonl"
DEFAULT_OUTPUT = RESULTS_DIR / "summary.md"

# ---------------------------------------------------------------------------
# Theme definitions for keyword-based grouping (REQ-026)
# Each entry: (theme_label, [keywords])
# Matching is case-insensitive substring search on the reason text.
# ---------------------------------------------------------------------------
YES_THEMES = [
    (
        "Affordability / Cost of Living",
        ["afford", "cost", "expensive", "price", "low income", "budget",
         "cheap", "saving", "wage", "worker"],
    ),
    (
        "Support for Small Restaurants",
        ["restaurant", "local", "small business", "neighborhood eatery",
         "owner", "establishment"],
    ),
    (
        "Fairness / Consumer Rights",
        ["fair", "consumer", "right", "protect", "exploitat", "gouging",
         "excessive fee", "cap", "limit"],
    ),
    (
        "Access / Equity",
        ["access", "equity", "equit", "underserved", "community",
         "delivery worker", "gig worker"],
    ),
    (
        "Economic Pressure",
        ["economic", "financial", "pressure", "struggle", "hardship",
         "survive", "margin"],
    ),
    ("Other", []),  # catch-all — always matches
]

NO_THEMES = [
    (
        "Free Market / Competition",
        ["free market", "competition", "competi", "market force", "natural",
         "innovation", "private"],
    ),
    (
        "Business Viability / App Quality",
        ["app", "service", "quality", "viable", "sustain", "revenue",
         "platform", "operat", "feature"],
    ),
    (
        "Job Losses / Driver Earnings",
        ["job", "driver", "earning", "gig", "lay off", "layoff",
         "employment", "worker pay", "reduce pay"],
    ),
    (
        "Government Overreach",
        ["government", "overreach", "interfer", "regulation",
         "over-regulat", "mandate"],
    ),
    (
        "Unintended Consequences",
        ["unintend", "consequence", "backfire", "result", "effect",
         "higher price", "less selection"],
    ),
    ("Other", []),  # catch-all
]


# Reasoning model used for LLM-based theme clustering (see analyze()).
# Routed through OpenRouter; uses the shared default model.
LLM_MODEL = llm.DEFAULT_MODEL


def _classify_reason(reason: str, themes: list[tuple]) -> str:
    """Return the first theme label whose keywords appear in the reason text."""
    lower = reason.lower()
    for label, keywords in themes:
        if not keywords:
            return label  # catch-all
        if any(kw in lower for kw in keywords):
            return label
    return "Other"


def _load_agents(agents_path: Path) -> list[dict]:
    if not agents_path.exists():
        raise FileNotFoundError(
            f"Agents file not found: {agents_path}\n"
            "Run generate_population.py first."
        )
    with open(agents_path, encoding="utf-8") as f:
        return json.load(f)


def _load_responses(responses_path: Path) -> list[dict]:
    if not responses_path.exists():
        raise FileNotFoundError(
            f"Responses file not found: {responses_path}\n"
            "Run run_scenario.py first."
        )
    records = []
    with open(responses_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Skipping malformed JSONL line: {exc}", file=sys.stderr)
    return records


def _join(agents: list[dict], responses: list[dict]) -> list[dict]:
    """Join agents with responses on id. Returns merged list in agent order."""
    response_map = {int(r["id"]): r for r in responses}
    joined = []
    for agent in agents:
        aid = int(agent["id"])
        resp = response_map.get(aid, {})
        joined.append({**agent, **resp})
    return joined


def _group_themes(
    joined: list[dict], vote_value: str, theme_list: list[tuple]
) -> list[tuple[str, list[str]]]:
    """
    For agents who voted `vote_value`, group reasons into themes.
    Returns top-3 themes as list of (theme_label, [agent_names]).
    """
    theme_agents: dict[str, list[str]] = defaultdict(list)
    for record in joined:
        if record.get("vote") != vote_value:
            continue
        reason = record.get("reason") or ""
        if reason in ("parse_error", "") or reason.startswith("api_error"):
            continue
        theme = _classify_reason(reason, theme_list)
        theme_agents[theme].append(record["name"])

    # Sort by count descending
    sorted_themes = sorted(theme_agents.items(), key=lambda x: len(x[1]), reverse=True)
    return sorted_themes[:3]


def _collect_reasons(joined: list[dict], vote_value: str) -> list[tuple[str, str]]:
    """Return [(agent_name, reason)] for valid votes matching vote_value."""
    out = []
    for record in joined:
        if record.get("vote") != vote_value:
            continue
        reason = record.get("reason") or ""
        if reason in ("parse_error", "") or reason.startswith("api_error"):
            continue
        out.append((record["name"], reason))
    return out


def _group_themes_llm(
    joined: list[dict], vote_value: str, question: str, client
) -> list[tuple[str, list[str]]]:
    """
    Cluster the reasons of agents who voted `vote_value` into up to 3 themes using
    the reasoning model, so labels fit the actual question. Returns top-3 (label, [names]).
    Raises on API/parse error so the caller can fall back to keyword grouping.
    """
    reasons = _collect_reasons(joined, vote_value)
    if not reasons:
        return []

    numbered = "\n".join(f"{i}. {reason}" for i, (_, reason) in enumerate(reasons))
    developer_prompt = (
        "You are a survey analyst grouping voters' free-text reasons into themes. "
        "Produce at most 3 concise theme labels (2-5 words each) that capture the "
        "main reasons given. Assign every reason to exactly one theme. "
        "Respond ONLY with valid JSON."
    )
    user_prompt = (
        f'Ballot question: "{question}"\n\n'
        f'The following voters all chose "{vote_value}". Their reasons:\n'
        f"{numbered}\n\n"
        'Respond in JSON: {"themes": [{"label": "<short theme name>", '
        '"indices": [<reason numbers belonging to this theme>]}]}. '
        "Use at most 3 themes, ordered from most to least common. Every number "
        f"from 0 to {len(reasons) - 1} must appear in exactly one theme."
    )

    content = llm.chat_json(client, LLM_MODEL, developer_prompt, user_prompt, effort="low")
    data = json.loads(content)

    themes: list[tuple[str, list[str]]] = []
    seen: set[int] = set()
    for entry in data.get("themes", []):
        label = str(entry.get("label", "")).strip() or "Other"
        names = []
        for idx in entry.get("indices", []):
            if isinstance(idx, int) and 0 <= idx < len(reasons) and idx not in seen:
                seen.add(idx)
                names.append(reasons[idx][0])
        if names:
            themes.append((label, names))

    # Any reasons the model failed to assign go into a catch-all so counts stay whole.
    leftover = [reasons[i][0] for i in range(len(reasons)) if i not in seen]
    if leftover:
        themes.append(("Other", leftover))

    themes.sort(key=lambda x: len(x[1]), reverse=True)
    return themes[:3]


def _themes_for(
    joined: list[dict],
    vote_value: str,
    keyword_themes: list[tuple],
    question: Optional[str],
    client,
) -> list[tuple[str, list[str]]]:
    """LLM theme grouping when a client + question are available; else keyword fallback."""
    if client is not None and question:
        try:
            return _group_themes_llm(joined, vote_value, question, client)
        except Exception as exc:  # noqa: BLE001 — any failure falls back gracefully
            print(
                f"  [WARN] LLM theme grouping failed ({exc}); using keyword fallback.",
                file=sys.stderr,
            )
    return _group_themes(joined, vote_value, keyword_themes)


def _valid_responses(joined: list[dict]) -> list[dict]:
    """Records with a real Yes/No vote and a non-error reason."""
    return [
        r for r in joined
        if r.get("reason")
        and r["reason"] not in ("parse_error", "")
        and not r["reason"].startswith("api_error")
        and r.get("vote") in ("Yes", "No")
    ]


def _most_persuasive_across_aisle(valid: list[dict], question: str, client) -> dict:
    """
    Ask the reasoning model to pick the single most persuasive-across-the-aisle response.

    'Most interesting' here means the response most likely to change the mind of a
    voter who chose the OPPOSITE side — the argument that best bridges the divide,
    not merely the loudest one for its own camp. Returns a shallow copy of the
    chosen record with an LLM-authored ``selection_reason``. Raises on API/parse
    error so the caller can fall back to the deterministic heuristic.
    """
    numbered = "\n".join(
        f"{i}. [{r['vote']}] {r['name']} ({r.get('occupation', 'n/a')}, "
        f"{r.get('neighborhood', 'n/a')}): {r['reason']}"
        for i, r in enumerate(valid)
    )
    developer_prompt = (
        "You are judging which single vote is the most persuasive ACROSS THE AISLE "
        "— the one response most likely to change the mind of a voter who chose the "
        "OPPOSITE side. Reward reasoning that engages the other side's concerns, is "
        "credible and specific, and finds common ground — not reasoning that merely "
        "rallies its own camp or restates a slogan. Respond ONLY with valid JSON."
    )
    user_prompt = (
        f'Ballot question: "{question}"\n\n'
        f"Panel responses (index, [vote], name, profile, reason):\n{numbered}\n\n"
        "Identify the ONE response most likely to move a voter from the other side. "
        'Respond in JSON: {"index": <chosen response number>, "reason": '
        '"<one sentence, max 30 words, naming the opposite side it would sway and '
        'why its argument lands with them>"}.'
    )

    content = llm.chat_json(client, LLM_MODEL, developer_prompt, user_prompt, effort="low")
    data = json.loads(content)

    idx = data.get("index")
    if not isinstance(idx, int) or not (0 <= idx < len(valid)):
        raise ValueError(f"LLM returned out-of-range index: {idx!r}")
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise ValueError("LLM returned an empty selection reason")

    chosen = dict(valid[idx])
    chosen["selection_reason"] = reason
    return chosen


def _most_interesting_longest(valid: list[dict]) -> dict:
    """Deterministic fallback: the response with the longest reason by word count."""
    winner = max(valid, key=lambda r: len(r["reason"].split()))
    word_count = len(winner["reason"].split())
    chosen = dict(winner)
    chosen["selection_reason"] = (
        f"Selected as the most elaborated response: at {word_count} words it "
        f"gives the longest, most detailed reasoning of the {len(valid)} valid "
        f"votes cast."
    )
    return chosen


def _most_interesting(
    joined: list[dict],
    question: Optional[str] = None,
    client=None,
) -> Optional[dict]:
    """
    Select the single most interesting response. REQ-025.

    When an OpenAI client and question are available AND both sides voted, the reasoning model
    picks the response most likely to change an opposing voter's mind ("most
    persuasive across the aisle") and authors the ``selection_reason``. When
    everyone voted the same way there is no aisle to cross, and on any LLM failure,
    it falls back to the deterministic longest-reason heuristic. Returns a shallow
    copy of the chosen record (so the added ``selection_reason`` never leaks into
    the full agent table), or None if no valid responses exist.
    """
    valid = _valid_responses(joined)
    if not valid:
        return None
    both_sides = (
        any(r["vote"] == "Yes" for r in valid)
        and any(r["vote"] == "No" for r in valid)
    )
    if client is not None and question and both_sides:
        try:
            return _most_persuasive_across_aisle(valid, question, client)
        except Exception as exc:  # noqa: BLE001 — any failure falls back gracefully
            print(
                f"  [WARN] LLM cross-aisle pick failed ({exc}); using longest-reason fallback.",
                file=sys.stderr,
            )
    return _most_interesting_longest(valid)


def _build_markdown(
    joined: list[dict],
    yes_count: int,
    no_count: int,
    null_count: int,
    yes_themes: list[tuple],
    no_themes: list[tuple],
    interesting: Optional[dict],
) -> str:
    """Assemble the full summary.md content."""
    total_valid = yes_count + no_count
    pct_yes = (yes_count / total_valid * 100) if total_valid > 0 else 0.0
    pct_no = (no_count / total_valid * 100) if total_valid > 0 else 0.0

    lines = []

    # ---- Section 1: Vote Tally ----
    lines.append("# SF Crowd Voting Simulator — Results Summary")
    lines.append("")
    lines.append("## 1. Vote Tally")
    lines.append("")
    lines.append(f"| Outcome | Count | % of Valid Votes |")
    lines.append(f"|---------|-------|-----------------|")
    lines.append(f"| Yes     | {yes_count}     | {pct_yes:.1f}%           |")
    lines.append(f"| No      | {no_count}     | {pct_no:.1f}%           |")
    lines.append(f"| Error   | {null_count}     | —               |")
    lines.append(f"| **Total** | **{yes_count + no_count + null_count}** | |")
    lines.append("")

    # ---- Section 2: Top Yes Reasons ----
    lines.append("## 2. Top Reasons — Yes")
    lines.append("")
    if yes_themes:
        for rank, (theme, names) in enumerate(yes_themes, start=1):
            agents_str = ", ".join(names[:3])
            more = f" (+{len(names) - 3} more)" if len(names) > 3 else ""
            lines.append(f"**{rank}. {theme}** ({len(names)} voters)")
            lines.append(f"  - Representatives: {agents_str}{more}")
            lines.append("")
    else:
        lines.append("_No Yes votes recorded._")
        lines.append("")

    # ---- Section 3: Top No Reasons ----
    lines.append("## 3. Top Reasons — No")
    lines.append("")
    if no_themes:
        for rank, (theme, names) in enumerate(no_themes, start=1):
            agents_str = ", ".join(names[:3])
            more = f" (+{len(names) - 3} more)" if len(names) > 3 else ""
            lines.append(f"**{rank}. {theme}** ({len(names)} voters)")
            lines.append(f"  - Representatives: {agents_str}{more}")
            lines.append("")
    else:
        lines.append("_No No votes recorded._")
        lines.append("")

    # ---- Section 4: Most Interesting Response ----
    lines.append("## 4. Most Interesting Response")
    lines.append("")
    if interesting:
        lines.append(
            f"> **{interesting['name']}**, age {interesting['age']}, "
            f"{interesting['neighborhood']} — {interesting['occupation']} "
            f"(Vote: **{interesting['vote']}**)"
        )
        lines.append(">")
        lines.append(f"> \"{interesting['reason']}\"")
        if interesting.get("selection_reason"):
            lines.append(">")
            lines.append(f"> _Why this one: {interesting['selection_reason']}_")
    else:
        lines.append("_No valid responses to display._")
    lines.append("")

    # ---- Section 5: Full Agent Table ----
    lines.append("## 5. Full Agent Table")
    lines.append("")
    lines.append(
        "| Name | Age | Neighborhood | Occupation | Household Income | Tenure | Vote |"
    )
    lines.append(
        "|------|-----|-------------|------------|-----------------:|--------|------|"
    )
    for record in joined:
        vote_display = record.get("vote") or "—"
        name = record.get("name", "")
        age = record.get("age", "")
        neighborhood = record.get("neighborhood", "")
        occupation = record.get("occupation", "")
        hh_income = record.get("household_income")
        income_display = f"${hh_income:,}/yr" if hh_income is not None else "N/A"
        tenure = record.get("tenure", "")
        lines.append(
            f"| {name} | {age} | {neighborhood} | {occupation} | "
            f"{income_display} | {tenure} | {vote_display} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core analyze function — importable by app.py
# ---------------------------------------------------------------------------

def analyze(agents: list[dict], responses: list[dict], client=None) -> dict:
    """
    Join agents and responses, compute analytics, write summary.md.
    Returns a summary dict for API consumption. REQ-024 through REQ-028.

    When `client` (an OpenAI client) is provided, Yes/No reasons are clustered into
    themes by the reasoning model so labels fit the asked question; otherwise (or on failure)
    the keyword-based buckets are used as a fallback.
    """
    joined = _join(agents, responses)

    yes_count = sum(1 for r in joined if r.get("vote") == "Yes")
    no_count = sum(1 for r in joined if r.get("vote") == "No")
    null_count = sum(1 for r in joined if r.get("vote") is None)

    # The question is stored on each response record (all share the same one).
    question = next((r.get("question") for r in responses if r.get("question")), None)

    yes_themes = _themes_for(joined, "Yes", YES_THEMES, question, client)
    no_themes = _themes_for(joined, "No", NO_THEMES, question, client)
    interesting = _most_interesting(joined, question, client)

    markdown = _build_markdown(
        joined, yes_count, no_count, null_count,
        yes_themes, no_themes, interesting
    )

    # Write summary.md — REQ-025
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUTPUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(markdown)

    return {
        "yes_count": yes_count,
        "no_count": no_count,
        "null_count": null_count,
        "pct_yes": round(yes_count / (yes_count + no_count) * 100, 1) if (yes_count + no_count) > 0 else 0.0,
        "yes_themes": [{"theme": t, "agents": a} for t, a in yes_themes],
        "no_themes": [{"theme": t, "agents": a} for t, a in no_themes],
        "most_interesting": interesting,
        "agents": joined,
        "markdown": markdown,
    }


# ---------------------------------------------------------------------------
# CLI entry point — REQ-028
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze voting results and produce summary.md."
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=str(DEFAULT_AGENTS),
        help=f"Path to agents.json (default: {DEFAULT_AGENTS})",
    )
    parser.add_argument(
        "--responses",
        type=str,
        default=str(DEFAULT_RESPONSES),
        help=f"Path to responses.jsonl (default: {DEFAULT_RESPONSES})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output path for summary.md (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        agents = _load_agents(Path(args.agents))
        responses = _load_responses(Path(args.responses))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    result = analyze(agents, responses)

    # REQ-027: Exit code 0 if all 30 non-null, 1 if any null
    if result["null_count"] > 0:
        failed = [
            r for r in result["agents"]
            if r.get("vote") is None
        ]
        print(
            f"WARNING: {result['null_count']} agent(s) have null votes:",
            file=sys.stderr,
        )
        for f in failed:
            print(f"  - Agent {f['id']}: {f['name']}", file=sys.stderr)
        print(f"\nSummary written to {args.output}")
        sys.exit(1)
    else:
        print(f"Summary written to {args.output}")
        print(
            f"Results: {result['yes_count']} Yes, {result['no_count']} No "
            f"({result['pct_yes']:.1f}% Yes)"
        )
        sys.exit(0)
