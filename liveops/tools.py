"""The analyst's tools. The agent and the rules baseline use exactly the same ones.

Guardrails live here, in code, not in the prompt:
* SQL is read-only (read-only connection, query_only pragma, single SELECT/WITH, row cap, time cap).
* A "ship" decision is re-checked against the data: assignment must be healthy (no sample ratio
  mismatch), the primary metric must improve significantly, and no guardrail metric may get
  significantly worse beyond its tolerance. A rejected decision goes back to the agent with reasons.
* Rollouts above 50% need a human, so the agent can recommend at most a staged rollout.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time

from . import stats

METRICS = {
    # name: (kind, column prefix, good direction, description)
    "d1_retention": ("proportion", "d1", "higher", "share of players active on day 1 after install"),
    "d7_retention": ("proportion", "d7", "higher", "share of players active on day 7 after install"),
    # Per session, not per player: a variant that keeps players longer gives them more sessions in
    # which to crash, so "players with a crash" would rise even if the build is equally stable.
    "crash_rate": ("ratio", "crash", "lower", "crashes per session, days 0-6"),
    "iap_arpu_d7": ("mean", "iap", "higher", "in-app purchase revenue per player, days 0-6 (USD)"),
    "ad_arpu_d7": ("mean", "ad", "higher", "ad revenue per player, days 0-6 (USD)"),
    "arpu_d7": ("mean", "rev", "higher", "total revenue per player, days 0-6 (USD)"),
}
# Relative change in the bad direction that a guardrail metric may tolerate.
GUARDRAIL_TOLERANCE = {"iap_arpu_d7": 0.05, "ad_arpu_d7": 0.05, "arpu_d7": 0.05, "crash_rate": 0.10}
MAX_ROLLOUT_WITHOUT_HUMAN = 50
DECISIONS = ("ship", "hold", "investigate")

VARIANT_SQL = """
WITH cohort AS (
  SELECT player_id, variant FROM assignments WHERE experiment_id = ?
),
per_player AS (
  SELECT c.variant,
         MAX(a.day = 1) AS d1,
         MAX(a.day = 7) AS d7,
         SUM(CASE WHEN a.day < 7 THEN a.crashes ELSE 0 END) AS crashes,
         SUM(CASE WHEN a.day < 7 THEN a.sessions ELSE 0 END) AS sessions,
         SUM(CASE WHEN a.day < 7 THEN a.iap_revenue ELSE 0 END) AS iap,
         SUM(CASE WHEN a.day < 7 THEN a.ad_revenue ELSE 0 END) AS ad
  FROM cohort c LEFT JOIN activity a ON a.player_id = c.player_id
  GROUP BY c.player_id, c.variant
)
SELECT variant, COUNT(*) AS n,
       SUM(d1) AS d1, SUM(d7) AS d7,
       SUM(crashes) AS crash_y, SUM(crashes * crashes) AS crash_yy,
       SUM(sessions) AS crash_x, SUM(sessions * sessions) AS crash_xx,
       SUM(crashes * sessions) AS crash_xy,
       SUM(iap) AS iap_sum, SUM(iap * iap) AS iap_sq,
       SUM(ad) AS ad_sum, SUM(ad * ad) AS ad_sq,
       SUM(iap + ad) AS rev_sum, SUM((iap + ad) * (iap + ad)) AS rev_sq
FROM per_player GROUP BY variant
"""

FORBIDDEN_SQL = re.compile(r"\b(attach|detach|pragma|insert|update|delete|drop|create|alter|replace|"
                           r"vacuum|reindex|analyze|load_extension)\b", re.IGNORECASE)


class ToolError(Exception):
    """Returned to the model as an error tool_result."""


def check_sql(query: str) -> str:
    q = re.sub(r"--[^\n]*|/\*.*?\*/", " ", query, flags=re.S).strip().rstrip(";").strip()
    if ";" in q:
        raise ToolError("only one statement is allowed")
    if not re.match(r"(?is)^(select|with)\b", q):
        raise ToolError("only SELECT or WITH queries are allowed")
    if FORBIDDEN_SQL.search(q):
        raise ToolError(f"keyword not allowed: {FORBIDDEN_SQL.search(q).group(0)}")
    return q


class AnalysisTools:
    def __init__(self, db_path, max_rows: int = 200, sql_timeout_s: float = 5.0):
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self.con.execute("PRAGMA query_only = ON")
        self.max_rows = max_rows
        self.sql_timeout_s = sql_timeout_s
        self.decision: dict | None = None
        self.rejections: list[dict] = []
        self._cache: dict[str, dict] = {}

    # ---------- tools exposed to the agent ----------

    def list_experiments(self) -> dict:
        rows = self.con.execute("SELECT experiment_id, name, hypothesis, primary_metric, guardrails, "
                                "planned_treatment_share FROM experiments").fetchall()
        return {"experiments": [dict(zip(["experiment_id", "name", "hypothesis", "primary_metric",
                                          "guardrail_metrics", "planned_treatment_share"], r)) for r in rows],
                "metrics": {k: v[3] for k, v in METRICS.items()},
                "tables": "players(player_id, install_day, country, platform, channel); "
                          "assignments(experiment_id, player_id, variant); activity(player_id, day, "
                          "sessions, levels_completed, ad_impressions, ad_revenue, iap_revenue, crashes)"}

    def run_sql(self, query: str) -> dict:
        q = check_sql(query)
        deadline = time.monotonic() + self.sql_timeout_s
        self.con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = self.con.execute(q)
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.OperationalError as exc:
            raise ToolError(f"SQL error: {exc}") from exc
        finally:
            self.con.set_progress_handler(None, 0)
        cols = [d[0] for d in cur.description]
        return {"columns": cols, "rows": rows[: self.max_rows], "truncated": len(rows) > self.max_rows}

    def check_srm(self, experiment_id: str) -> dict:
        planned = self._experiment(experiment_id)["planned_treatment_share"]
        agg = self._aggregates(experiment_id)
        return stats.srm_test(agg["control"]["n"], agg["treatment"]["n"], planned)

    def ab_test(self, experiment_id: str, metric: str) -> dict:
        if metric not in METRICS:
            raise ToolError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
        kind, col, good, _ = METRICS[metric]
        agg = self._aggregates(experiment_id)
        c, t = agg["control"], agg["treatment"]
        if kind == "proportion":
            result = stats.proportion_test(metric, c[col], c["n"], t[col], t["n"])
        elif kind == "ratio":
            pick = lambda v: tuple(v[f"{col}_{k}"] for k in ("y", "yy", "x", "xx", "xy")) + (v["n"],)
            result = stats.ratio_test(metric, pick(c), pick(t))
        else:
            result = stats.mean_test(metric, c[f"{col}_sum"], c[f"{col}_sq"], c["n"],
                                     t[f"{col}_sum"], t[f"{col}_sq"], t["n"])
        out = result.to_dict()
        out["good_direction"] = good
        improved = result.abs_diff > 0 if good == "higher" else result.abs_diff < 0
        out["verdict"] = ("significant improvement" if result.significant and improved else
                          "significant regression" if result.significant else "no significant change")
        return out

    def submit_decision(self, experiment_id: str, decision: str, rationale: str,
                        rollout_percent: int = 0) -> dict:
        if decision not in DECISIONS:
            raise ToolError(f"decision must be one of {DECISIONS}")
        exp = self._experiment(experiment_id)
        violations = []
        if decision == "ship":
            srm = self.check_srm(experiment_id)
            if not srm["ok"]:
                violations.append(f"sample ratio mismatch (p={srm['p_value']:.2g}): fix assignment first")
            primary = self.ab_test(experiment_id, exp["primary_metric"])
            if primary["verdict"] != "significant improvement":
                violations.append(f"primary metric {exp['primary_metric']}: {primary['verdict']}")
            for metric in exp["guardrails"]:
                breach = self.guardrail_breach(experiment_id, metric)
                if breach:
                    violations.append(breach)
        if violations:
            rejection = {"accepted": False, "decision": decision, "violations": violations}
            self.rejections.append(rejection)
            return rejection
        flagged = decision == "ship" and rollout_percent > MAX_ROLLOUT_WITHOUT_HUMAN
        self.decision = {
            "accepted": True, "experiment_id": experiment_id, "decision": decision,
            "rationale": rationale,
            "rollout_percent": min(rollout_percent, MAX_ROLLOUT_WITHOUT_HUMAN) if decision == "ship" else 0,
            "needs_human_approval": flagged,
            "impact": self.impact(experiment_id) if decision == "ship" else None,
        }
        return self.decision

    # ---------- helpers (not exposed) ----------

    def guardrail_breach(self, experiment_id: str, metric: str) -> str | None:
        res = self.ab_test(experiment_id, metric)
        tolerance = GUARDRAIL_TOLERANCE.get(metric, 0.0)
        worse = res["rel_diff"] < -tolerance if res["good_direction"] == "higher" else res["rel_diff"] > tolerance
        if res["significant"] and worse:
            return (f"guardrail {metric} moved {100 * res['rel_diff']:+.1f}% "
                    f"(p={res['p_value']:.2g}, tolerance {100 * tolerance:.0f}%)")
        return None

    def impact(self, experiment_id: str, installs: int = 100_000) -> dict:
        """What a full rollout would mean per 100k installs, with 95% intervals."""
        d7 = self.ab_test(experiment_id, "d7_retention")
        rev = self.ab_test(experiment_id, "arpu_d7")
        return {
            "per_installs": installs,
            "extra_d7_players": [round(d7["ci_low"] * installs), round(d7["abs_diff"] * installs),
                                 round(d7["ci_high"] * installs)],
            "extra_7d_revenue_usd": [round(rev["ci_low"] * installs), round(rev["abs_diff"] * installs),
                                     round(rev["ci_high"] * installs)],
        }

    def _experiment(self, experiment_id: str) -> dict:
        row = self.con.execute("SELECT primary_metric, guardrails, planned_treatment_share "
                               "FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        if row is None:
            raise ToolError(f"unknown experiment {experiment_id!r}")
        return {"primary_metric": row[0], "guardrails": row[1].split(","), "planned_treatment_share": row[2]}

    def _aggregates(self, experiment_id: str) -> dict:
        if experiment_id not in self._cache:
            self._experiment(experiment_id)
            cur = self.con.execute(VARIANT_SQL, (experiment_id,))
            cols = [d[0] for d in cur.description]
            rows = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
            if set(rows) != {"control", "treatment"}:
                raise ToolError("experiment needs exactly a control and a treatment variant")
            self._cache[experiment_id] = rows
        return self._cache[experiment_id]


# JSON schemas for the Claude tools (strict: every property required, no extras).
TOOL_SPECS = [
    {"name": "list_experiments",
     "description": "List experiments with hypothesis, primary metric and guardrail metrics, plus the metric "
                    "definitions and table schema. Call this first.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "check_srm",
     "description": "Sample ratio mismatch check: is the control/treatment split what was planned? "
                    "If ok is false the experiment data cannot be trusted.",
     "input_schema": {"type": "object", "properties": {"experiment_id": {"type": "string"}},
                      "required": ["experiment_id"], "additionalProperties": False}},
    {"name": "ab_test",
     "description": "Compare treatment against control on one metric: means, absolute and relative difference, "
                    "95% confidence interval, p-value and a verdict.",
     "input_schema": {"type": "object", "properties": {
         "experiment_id": {"type": "string"},
         "metric": {"type": "string", "enum": sorted(METRICS)}},
         "required": ["experiment_id", "metric"], "additionalProperties": False}},
    {"name": "run_sql",
     "description": "Run one read-only SQLite SELECT (or WITH) query for segment analysis. At most 200 rows "
                    "are returned. Tables: players, assignments, activity, experiments.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                      "required": ["query"], "additionalProperties": False}},
    {"name": "submit_decision",
     "description": "Submit the final decision. ship = roll out the treatment; hold = keep control; "
                    "investigate = the data has a quality problem. A ship decision is re-checked against the "
                    "data and rejected with reasons if it fails a guardrail. Rollouts above 50% need a human.",
     "input_schema": {"type": "object", "properties": {
         "experiment_id": {"type": "string"},
         "decision": {"type": "string", "enum": list(DECISIONS)},
         "rationale": {"type": "string"},
         "rollout_percent": {"type": "integer"}},
         "required": ["experiment_id", "decision", "rationale", "rollout_percent"],
         "additionalProperties": False}},
]


def dispatch(tools: AnalysisTools, name: str, args: dict) -> tuple[dict, bool]:
    """Run one tool call. Returns (result, is_error)."""
    fn = {"list_experiments": tools.list_experiments, "check_srm": tools.check_srm,
          "ab_test": tools.ab_test, "run_sql": tools.run_sql,
          "submit_decision": tools.submit_decision}.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}, True
    try:
        return fn(**args), False
    except ToolError as exc:
        return {"error": str(exc)}, True
    except TypeError as exc:
        return {"error": f"bad arguments: {exc}"}, True


def compact(result: dict) -> str:
    # run_sql already caps rows and says so with "truncated"; everything else is small.
    return json.dumps(result, default=str)
