"""Offline tests: no API key needed. The Claude loop is driven by a scripted fake client."""
import math
import sqlite3
from types import SimpleNamespace

import pytest

from liveops import stats
from liveops.agent import Budget, run_agent
from liveops.policy import rules_analyst
from liveops.simulate import BY_NAME, simulate
from liveops.tools import AnalysisTools, ToolError, check_sql, dispatch
from liveops.trace import Tracer


@pytest.fixture(scope="module")
def dbs(tmp_path_factory):
    root = tmp_path_factory.mktemp("db")
    return {name: simulate(BY_NAME[name], root / f"{name}.db", seed=7)
            for name in ("easier_early_levels", "more_free_lives", "new_onboarding")}


def test_proportion_test_matches_hand_calculation():
    r = stats.proportion_test("m", 100, 1000, 130, 1000)
    pooled = 230 / 2000
    z = 0.03 / math.sqrt(pooled * (1 - pooled) * 2 / 1000)
    assert r.abs_diff == pytest.approx(0.03)
    assert r.p_value == pytest.approx(2 * (1 - stats.norm_cdf(z)))
    assert r.ci_low < 0.03 < r.ci_high and r.significant


def test_mean_test_from_sums_matches_direct_variance():
    a, b = [0, 0, 1, 2, 5, 0, 0, 3], [1, 0, 4, 2, 6, 0, 1, 3]
    r = stats.mean_test("m", sum(a), sum(x * x for x in a), len(a), sum(b), sum(x * x for x in b), len(b))
    var = lambda xs: sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1)
    se = math.sqrt(var(a) / len(a) + var(b) / len(b))
    assert r.abs_diff == pytest.approx(sum(b) / 8 - sum(a) / 8)
    assert r.ci_high - r.abs_diff == pytest.approx(stats.Z95 * se)


def test_ratio_test_delta_method_against_simulation():
    import random
    rng = random.Random(3)

    def arm(rate):
        rows = []
        for _ in range(4000):
            x = rng.randint(1, 8)                                   # sessions
            rows.append((sum(rng.random() < rate for _ in range(x)), x))
        ys, xs = [r[0] for r in rows], [r[1] for r in rows]
        return (sum(ys), sum(y * y for y in ys), sum(xs), sum(x * x for x in xs),
                sum(x * y for x, y in rows), len(rows))

    same = stats.ratio_test("c", arm(0.01), arm(0.01))
    worse = stats.ratio_test("c", arm(0.01), arm(0.03))
    assert not same.significant
    assert worse.significant and worse.rel_diff > 1.0


def test_srm_detects_imbalance():
    assert stats.srm_test(5000, 5000)["ok"]
    bad = stats.srm_test(5200, 4800)              # chi2 = 16
    assert bad["chi2"] == pytest.approx(16.0) and not bad["ok"]


def test_mde_shrinks_with_sample_size():
    assert stats.mde_proportion(0.15, 40_000) < stats.mde_proportion(0.15, 10_000)


@pytest.mark.parametrize("query", [
    "DELETE FROM players", "SELECT 1; DROP TABLE players", "PRAGMA table_info(players)",
    "ATTACH DATABASE 'x.db' AS x", "UPDATE activity SET iap_revenue = 0", "  /* hi */ INSERT INTO x VALUES (1)",
])
def test_sql_guard_rejects_writes_and_multi_statements(query):
    with pytest.raises(ToolError):
        check_sql(query)


def test_sql_guard_allows_reads(dbs):
    tools = AnalysisTools(dbs["easier_early_levels"], max_rows=5)
    out = tools.run_sql("-- segment check\nWITH x AS (SELECT country, COUNT(*) n FROM players GROUP BY country) "
                        "SELECT * FROM x ORDER BY n DESC;")
    assert out["columns"] == ["country", "n"] and len(out["rows"]) == 5
    big = tools.run_sql("SELECT * FROM activity")
    assert big["truncated"] and len(big["rows"]) == 5


def test_connection_is_read_only_even_without_the_guard(dbs):
    tools = AnalysisTools(dbs["easier_early_levels"])
    with pytest.raises(sqlite3.OperationalError):
        tools.con.execute("DELETE FROM players")


def test_ship_is_rejected_when_a_guardrail_breaks(dbs):
    tools = AnalysisTools(dbs["more_free_lives"])
    result = tools.submit_decision("exp_more_free_lives", "ship", "d7 up", 50)
    assert not result["accepted"]
    assert any("iap_arpu_d7" in v for v in result["violations"])
    assert tools.decision is None


def test_ship_is_rejected_on_sample_ratio_mismatch(dbs):
    tools = AnalysisTools(dbs["new_onboarding"])
    assert not tools.check_srm("exp_new_onboarding")["ok"]
    result = tools.submit_decision("exp_new_onboarding", "ship", "d7 up", 50)
    assert any("sample ratio mismatch" in v for v in result["violations"])


def test_good_ship_is_accepted_with_staged_rollout_and_impact(dbs):
    tools = AnalysisTools(dbs["easier_early_levels"])
    result = tools.submit_decision("exp_easier_early_levels", "ship", "d7 up", 100)
    assert result["accepted"] and result["rollout_percent"] == 50 and result["needs_human_approval"]
    low, mid, high = result["impact"]["extra_d7_players"]
    assert 0 < low < mid < high


def test_rules_analyst_decisions(dbs):
    expected = {"easier_early_levels": "ship", "more_free_lives": "hold", "new_onboarding": "investigate"}
    for name, want in expected.items():
        decision = rules_analyst(AnalysisTools(dbs[name]), f"exp_{name}", Tracer(None))
        assert decision["decision"] == want


def test_dispatch_turns_bad_input_into_tool_errors(dbs):
    tools = AnalysisTools(dbs["easier_early_levels"])
    assert dispatch(tools, "ab_test", {"experiment_id": "nope", "metric": "d7_retention"})[1]
    assert dispatch(tools, "ab_test", {"experiment_id": "exp_easier_early_levels", "metric": "ltv"})[1]
    assert dispatch(tools, "drop_tables", {})[1]


# ---------- the Claude loop, with a scripted stand-in for the API ----------

def tool_use(i, name, **args):
    return SimpleNamespace(type="tool_use", id=f"tu_{i}", name=name, input=args)


def reply(*blocks, stop="tool_use"):
    return SimpleNamespace(stop_reason=stop, content=list(blocks),
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=200))


class ScriptedClient:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def test_agent_recovers_from_a_blocked_ship(dbs):
    exp = "exp_more_free_lives"
    client = ScriptedClient([
        reply(tool_use(1, "list_experiments")),
        reply(tool_use(2, "ab_test", experiment_id=exp, metric="d7_retention")),
        reply(tool_use(3, "submit_decision", experiment_id=exp, decision="ship",
                       rationale="d7 up", rollout_percent=50)),
        reply(tool_use(4, "submit_decision", experiment_id=exp, decision="hold",
                       rationale="IAP guardrail breached", rollout_percent=0)),
    ])
    tracer = Tracer(None)
    decision = run_agent(AnalysisTools(dbs["more_free_lives"]), exp, tracer, client=client)
    assert decision["decision"] == "hold" and not decision.get("fail_closed")
    s = tracer.summary()
    assert s["guardrail_rejections"] == 1 and s["llm_calls"] == 4 and s["input_tokens"] == 4000
    # The rejection reached the model as a tool result, and tools were sent with strict schemas.
    results = [block for m in client.requests[-1]["messages"] if m["role"] == "user"
               and isinstance(m["content"], list) for block in m["content"]]
    rejected = next(b for b in results if b["tool_use_id"] == "tu_3")
    assert "violations" in rejected["content"] and not rejected["is_error"]
    assert all(t["strict"] for t in client.requests[0]["tools"])


def test_agent_fails_closed_on_refusal(dbs):
    client = ScriptedClient([reply(stop="refusal")])
    decision = run_agent(AnalysisTools(dbs["easier_early_levels"]), "exp_easier_early_levels",
                         Tracer(None), client=client)
    assert decision["decision"] == "hold" and decision["fail_closed"]


def test_agent_fails_closed_when_budget_runs_out(dbs):
    client = ScriptedClient([reply(tool_use(i, "list_experiments")) for i in range(5)])
    decision = run_agent(AnalysisTools(dbs["easier_early_levels"]), "exp_easier_early_levels",
                         Tracer(None), client=client, budget=Budget(max_turns=3))
    assert decision["fail_closed"] and len(client.requests) == 3
