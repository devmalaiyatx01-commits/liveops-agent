"""Command line: python -m liveops {simulate,analyze,eval,trace,scenarios}"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .simulate import BY_NAME, SCENARIOS, simulate


def cmd_scenarios(args) -> int:
    for s in SCENARIOS:
        print(f"{s.name:22} expected={s.expected:11} {s.hypothesis}")
    return 0


def cmd_simulate(args) -> int:
    path = simulate(BY_NAME[args.scenario], args.db, args.seed)
    print(f"wrote {path} (experiment {BY_NAME[args.scenario].experiment_id})")
    return 0


def cmd_analyze(args) -> int:
    from .agent import run_agent
    from .policy import rules_analyst
    from .tools import AnalysisTools
    from .trace import Tracer

    tools = AnalysisTools(args.db)
    experiment = args.experiment or tools.list_experiments()["experiments"][0]["experiment_id"]
    tracer = Tracer(args.runs)
    if args.policy == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set; use --policy rules or set a key", file=sys.stderr)
            return 1
        decision = run_agent(tools, experiment, tracer)
    else:
        decision = rules_analyst(tools, experiment, tracer)
    print(json.dumps(decision, indent=2))
    print(json.dumps(tracer.summary()))
    if tracer.path:
        print(f"trace: {tracer.path}")
    return 0


def cmd_eval(args) -> int:
    from .evals import report, run_suite

    if args.policy == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1
    rows = run_suite(args.policy, tuple(range(1, args.seeds + 1)), trace_dir=args.runs)
    text = report(rows, args.policy)
    print(text)
    if args.write:
        with open(args.write, "w", encoding="utf-8") as fh:
            fh.write("# Scenario evals\n\n" + text)
    return 0


def cmd_trace(args) -> int:
    from .trace import show
    print(show(args.path))
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="liveops")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scenarios", help="list planted-truth scenarios").set_defaults(func=cmd_scenarios)

    p = sub.add_parser("simulate", help="generate a game telemetry database for one scenario")
    p.add_argument("--scenario", choices=sorted(BY_NAME), default="easier_early_levels")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--db", default="data/game.db")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("analyze", help="decide one experiment")
    p.add_argument("--db", default="data/game.db")
    p.add_argument("--experiment")
    p.add_argument("--policy", choices=["rules", "claude"], default="rules")
    p.add_argument("--runs", default="runs", help="directory for JSONL traces")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("eval", help="run every scenario with several seeds")
    p.add_argument("--policy", choices=["rules", "naive", "claude"], default="rules")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--runs", default=None, help="keep traces in this directory")
    p.add_argument("--write", help="also write the report to this Markdown file")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("trace", help="print a trace as a timeline")
    p.add_argument("path")
    p.set_defaults(func=cmd_trace)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
