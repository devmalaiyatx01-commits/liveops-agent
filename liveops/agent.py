"""Claude as the experiment analyst: a tool-use loop with budgets, tracing and a fail-closed default.

The loop runs on the host because the tools read a local SQLite database and the guardrails must
be enforced in our code. If the agent runs out of budget, is refused, or never gets a decision
accepted, the result is "hold": doing nothing is the safe default for a live game.
"""
from __future__ import annotations

import time

from .tools import TOOL_SPECS, AnalysisTools, compact, dispatch
from .trace import Tracer

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are the LiveOps analyst for a casual mobile puzzle game. You decide whether an A/B experiment should ship.

Work like a careful analyst:
1. Read the experiment (list_experiments) to learn its hypothesis, primary metric and guardrail metrics.
2. Check that assignment is healthy (check_srm). If there is a sample ratio mismatch, the comparison is biased: the decision is investigate, whatever the metrics say.
3. Test the primary metric, then every guardrail metric (ab_test). Use run_sql only when a segment question would change the decision.
4. Submit one decision with submit_decision:
   - ship: the primary metric improved significantly and no guardrail regressed beyond tolerance. Recommend a staged rollout of at most 50%.
   - hold: no significant improvement, a regression, or a guardrail breach.
   - investigate: the data itself is not trustworthy.
   Quote the numbers (difference, confidence interval, p-value) in the rationale.

A ship decision is re-checked against the data. If it is rejected, read the violations and decide again. Never claim a result you did not get from a tool."""


class Budget:
    def __init__(self, max_turns: int = 10, max_tool_calls: int = 24, max_input_tokens: int = 400_000):
        self.max_turns, self.max_tool_calls, self.max_input_tokens = max_turns, max_tool_calls, max_input_tokens
        self.turns = self.tool_calls = self.input_tokens = 0

    def exceeded(self) -> str | None:
        if self.turns >= self.max_turns:
            return f"turn budget ({self.max_turns})"
        if self.tool_calls >= self.max_tool_calls:
            return f"tool-call budget ({self.max_tool_calls})"
        if self.input_tokens >= self.max_input_tokens:
            return f"input-token budget ({self.max_input_tokens})"
        return None


def run_agent(tools: AnalysisTools, experiment_id: str, tracer: Tracer, client=None,
              model: str = MODEL, budget: Budget | None = None) -> dict:
    if client is None:
        import anthropic  # optional dependency, only needed for the live agent
        client = anthropic.Anthropic()
    budget = budget or Budget()
    specs = [{**spec, "strict": True} for spec in TOOL_SPECS]
    messages = [{"role": "user", "content": f"Analyse experiment {experiment_id} and submit your decision."}]
    nudged = False

    while not (tools.decision and tools.decision.get("accepted")):
        reason = budget.exceeded()
        if reason:
            tracer.event("budget", "exhausted", detail=reason)
            break
        start = time.monotonic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            tools=specs,
            output_config={"effort": "medium"},
            messages=messages,
        )
        budget.turns += 1
        usage = getattr(response, "usage", None)
        tin = getattr(usage, "input_tokens", 0) or 0
        tout = getattr(usage, "output_tokens", 0) or 0
        budget.input_tokens += tin
        tracer.event("llm_call", model, input_tokens=tin, output_tokens=tout,
                     stop_reason=response.stop_reason, latency_ms=int((time.monotonic() - start) * 1000))

        if response.stop_reason == "refusal":
            tracer.event("budget", "refused", detail=getattr(response, "stop_details", None))
            break
        # History is append-only: the assistant turn goes back exactly as received.
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": "You have not had a decision accepted yet. "
                                                        "Call submit_decision."})
            continue

        results = []
        for block in tool_uses:
            budget.tool_calls += 1
            result, is_error = dispatch(tools, block.name, dict(block.input))
            tracer.event("tool_call", block.name, input=block.input, output=result, is_error=is_error)
            if block.name == "submit_decision" and not is_error and not result.get("accepted", True):
                tracer.event("guardrail", "ship_blocked", detail=result["violations"])
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": compact(result), "is_error": is_error})
        messages.append({"role": "user", "content": results})

    if tools.decision and tools.decision.get("accepted"):
        tracer.event("decision", tools.decision["decision"], output=tools.decision)
        return tools.decision
    fallback = {"accepted": True, "experiment_id": experiment_id, "decision": "hold", "fail_closed": True,
                "rationale": "Agent did not reach an accepted decision; defaulting to hold.",
                "rollout_percent": 0, "needs_human_approval": True, "impact": None}
    tracer.event("decision", "hold (fail closed)", output=fallback)
    return fallback
