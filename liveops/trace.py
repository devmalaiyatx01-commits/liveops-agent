"""Observability: every model call, tool call and guardrail event goes to a JSONL trace."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

# Claude Opus 5.5 list price, USD per million tokens (input, output).
PRICE_PER_MTOK = (4.0, 20.0)


class Tracer:
    def __init__(self, out_dir: str | Path | None = "runs", run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = Path(out_dir, f"{self.run_id}.jsonl") if out_dir else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.started = time.monotonic()

    def event(self, kind: str, name: str, **fields) -> dict:
        ev = {"run_id": self.run_id, "step": len(self.events) + 1,
              "t_ms": int((time.monotonic() - self.started) * 1000), "kind": kind, "name": name, **fields}
        self.events.append(ev)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, default=str) + "\n")
        return ev

    def summary(self) -> dict:
        llm = [e for e in self.events if e["kind"] == "llm_call"]
        tools = [e for e in self.events if e["kind"] == "tool_call"]
        tin = sum(e.get("input_tokens", 0) for e in llm)
        tout = sum(e.get("output_tokens", 0) for e in llm)
        return {
            "run_id": self.run_id,
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for e in tools if e.get("is_error")),
            "guardrail_rejections": sum(1 for e in self.events if e["kind"] == "guardrail"),
            "input_tokens": tin,
            "output_tokens": tout,
            "cost_usd": round(tin / 1e6 * PRICE_PER_MTOK[0] + tout / 1e6 * PRICE_PER_MTOK[1], 4),
            "wall_ms": int((time.monotonic() - self.started) * 1000),
        }


def show(path: str | Path) -> str:
    """Readable timeline of a trace file."""
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        e = json.loads(raw)
        extra = ""
        if e["kind"] == "llm_call":
            extra = f"in={e.get('input_tokens')} out={e.get('output_tokens')} stop={e.get('stop_reason')}"
        elif e["kind"] == "tool_call":
            extra = f"{json.dumps(e.get('input'))[:80]} -> {str(e.get('output'))[:90]}"
        elif e["kind"] in ("guardrail", "decision", "budget"):
            extra = str(e.get("detail") or e.get("output"))[:160]
        lines.append(f"{e['step']:>3} {e['t_ms']:>7}ms {e['kind']:<10} {e['name']:<18} {extra}")
    return "\n".join(lines)
