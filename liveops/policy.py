"""Deterministic analyst: the baseline the agent is compared with, and the offline mode.

It follows the playbook a careful analyst would: trust check first, then the primary metric,
then guardrails. It calls the same tools as the agent, so it is traced the same way.
"""
from __future__ import annotations

from .tools import AnalysisTools, dispatch
from .trace import Tracer


def call(tools: AnalysisTools, tracer: Tracer, name: str, **args) -> dict:
    result, is_error = dispatch(tools, name, args)
    tracer.event("tool_call", name, input=args, output=result, is_error=is_error)
    if name == "submit_decision" and not result.get("accepted", True):
        tracer.event("guardrail", "ship_blocked", detail=result["violations"])
    return result


def naive_analyst(tools: AnalysisTools, experiment_id: str, tracer: Tracer) -> dict:
    """Looks only at the primary metric. Exists to show what the code guardrails catch."""
    exp = next(e for e in call(tools, tracer, "list_experiments")["experiments"]
               if e["experiment_id"] == experiment_id)
    primary = call(tools, tracer, "ab_test", experiment_id=experiment_id, metric=exp["primary_metric"])
    if primary["verdict"] == "significant improvement":
        decision = call(tools, tracer, "submit_decision", experiment_id=experiment_id, decision="ship",
                        rationale="Primary metric improved.", rollout_percent=100)
        if decision.get("accepted"):
            return decision
    return call(tools, tracer, "submit_decision", experiment_id=experiment_id, decision="hold",
                rationale="No accepted ship.", rollout_percent=0)


def rules_analyst(tools: AnalysisTools, experiment_id: str, tracer: Tracer) -> dict:
    exp = next(e for e in call(tools, tracer, "list_experiments")["experiments"]
               if e["experiment_id"] == experiment_id)
    srm = call(tools, tracer, "check_srm", experiment_id=experiment_id)
    if not srm["ok"]:
        return call(tools, tracer, "submit_decision", experiment_id=experiment_id, decision="investigate",
                    rationale=f"Sample ratio mismatch: treatment share {srm['observed_treatment_share']} "
                              f"vs planned {srm['expected_treatment_share']} (p={srm['p_value']:.2g}).",
                    rollout_percent=0)
    primary = call(tools, tracer, "ab_test", experiment_id=experiment_id, metric=exp["primary_metric"])
    guard = {m: call(tools, tracer, "ab_test", experiment_id=experiment_id, metric=m)
             for m in exp["guardrail_metrics"].split(",")}
    regressions = [m for m, r in guard.items() if r["verdict"] == "significant regression"]
    if primary["verdict"] == "significant improvement" and not regressions:
        decision = call(tools, tracer, "submit_decision", experiment_id=experiment_id, decision="ship",
                        rationale=f"{exp['primary_metric']} {100 * primary['abs_diff']:+.2f} pp "
                                  f"(p={primary['p_value']:.2g}); guardrails within tolerance.",
                        rollout_percent=50)
        if decision.get("accepted"):
            return decision
    reason = (f"{exp['primary_metric']}: {primary['verdict']}"
              + (f"; regressions in {', '.join(regressions)}" if regressions else ""))
    return call(tools, tracer, "submit_decision", experiment_id=experiment_id, decision="hold",
                rationale=reason, rollout_percent=0)
