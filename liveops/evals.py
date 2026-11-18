"""Scenario evals: does the analyst make the decision the planted truth calls for?

Each scenario is simulated with several seeds. The headline numbers are decision accuracy and
false ships (shipping something that should not ship), the expensive mistake in a live game.
"""
from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path

from .agent import run_agent
from .policy import naive_analyst, rules_analyst
from .simulate import SCENARIOS, simulate
from .tools import AnalysisTools
from .trace import Tracer


def run_case(scenario, seed: int, policy: str, client=None, trace_dir=None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db = simulate(scenario, Path(tmp) / "game.db", seed)
        tools = AnalysisTools(db)
        tracer = Tracer(trace_dir, run_id=f"{policy}-{scenario.name}-s{seed}")
        start = time.monotonic()
        if policy == "rules":
            decision = rules_analyst(tools, scenario.experiment_id, tracer)
        elif policy == "naive":
            decision = naive_analyst(tools, scenario.experiment_id, tracer)
        else:
            decision = run_agent(tools, scenario.experiment_id, tracer, client=client)
        tools.con.close()
    summary = tracer.summary()
    return {"scenario": scenario.name, "seed": seed, "expected": scenario.expected,
            "decision": decision["decision"], "correct": decision["decision"] == scenario.expected,
            "fail_closed": bool(decision.get("fail_closed")),
            "seconds": round(time.monotonic() - start, 1), **summary}


def run_suite(policy: str = "rules", seeds=(1, 2, 3), client=None, trace_dir=None) -> list[dict]:
    return [run_case(s, seed, policy, client, trace_dir) for s in SCENARIOS for seed in seeds]


def report(rows: list[dict], policy: str) -> str:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    false_ship = sum(1 for r in rows if r["decision"] == "ship" and r["expected"] != "ship")
    missed_ship = sum(1 for r in rows if r["expected"] == "ship" and r["decision"] != "ship")
    lines = [f"## Policy: {policy}", "",
             f"Decision accuracy: **{correct}/{n}**. False ships: **{false_ship}**. Missed ships: {missed_ship}. "
             f"Guardrail rejections: {sum(r['guardrail_rejections'] for r in rows)}. "
             f"Fail-closed runs: {sum(r['fail_closed'] for r in rows)}.", "",
             "| Scenario | Expected | Decisions (by seed) | Correct | Avg tool calls | Avg cost (USD) |",
             "|---|---|---|---:|---:|---:|"]
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append(r)
    for name, rs in by.items():
        lines.append(f"| {name} | {rs[0]['expected']} | {', '.join(r['decision'] for r in rs)} | "
                     f"{sum(r['correct'] for r in rs)}/{len(rs)} | "
                     f"{sum(r['tool_calls'] for r in rs) / len(rs):.1f} | "
                     f"{sum(r['cost_usd'] for r in rs) / len(rs):.3f} |")
    confusion = Counter((r["expected"], r["decision"]) for r in rows)
    lines += ["", "Confusion (expected -> decided): " +
              ", ".join(f"{e}->{d}: {c}" for (e, d), c in sorted(confusion.items()))]
    return "\n".join(lines) + "\n"
