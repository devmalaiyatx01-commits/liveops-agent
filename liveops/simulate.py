"""Synthetic telemetry for a casual puzzle game running one A/B experiment.

Each scenario plants a known truth (a real retention lift, no effect, a revenue regression,
a broken assignment, a crash regression) so the analyst's decision can be scored.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DAYS = 14
# Share of players active on day d after install, before player-level variation.
BASE_ACTIVE = np.array([1.0, 0.40, 0.31, 0.26, 0.23, 0.21, 0.19, 0.17, 0.16, 0.15, 0.14, 0.13, 0.125, 0.12])
COUNTRIES = (["IN", "US", "BR", "ID", "DE"], [0.35, 0.2, 0.2, 0.15, 0.1])
CHANNELS = (["organic", "meta_ads", "google_uac", "unity_ads"], [0.4, 0.25, 0.2, 0.15])
PRICE_POINTS = (np.array([0.99, 4.99, 9.99, 19.99]), [0.55, 0.3, 0.1, 0.05])

SCHEMA = """
CREATE TABLE players (player_id INTEGER PRIMARY KEY, install_day INTEGER, country TEXT,
                      platform TEXT, channel TEXT);
CREATE TABLE experiments (experiment_id TEXT PRIMARY KEY, name TEXT, hypothesis TEXT,
                          primary_metric TEXT, guardrails TEXT, planned_treatment_share REAL);
CREATE TABLE assignments (experiment_id TEXT, player_id INTEGER, variant TEXT);
CREATE TABLE activity (player_id INTEGER, day INTEGER, sessions INTEGER, levels_completed INTEGER,
                       ad_impressions INTEGER, ad_revenue REAL, iap_revenue REAL, crashes INTEGER);
CREATE INDEX idx_activity_player ON activity(player_id, day);
CREATE INDEX idx_assign ON assignments(experiment_id, variant);
"""


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    hypothesis: str
    expected: str                    # ship | hold | investigate
    d1_lift: float = 0.0             # absolute change in day-1 activity for treatment
    d7_lift: float = 0.0             # absolute change from day 7 onwards
    iap_mult: float = 1.0            # multiplier on purchase probability
    ad_mult: float = 1.0             # multiplier on ad impressions
    crash_mult: float = 1.0          # multiplier on crash rate
    drop_low_engaged_treatment: float = 0.0   # assignment bug: share of low-engaged treatment users lost
    n_players: int = 20_000

    @property
    def experiment_id(self) -> str:
        return f"exp_{self.name}"


SCENARIOS = [
    Scenario("easier_early_levels", "Easier levels 5-15",
             "Lowering difficulty on levels 5-15 reduces early churn and raises D7 retention.",
             "ship", d1_lift=0.03, d7_lift=0.02),
    Scenario("reward_chest", "Daily reward chest",
             "A daily reward chest brings players back on day 7.", "hold"),
    Scenario("more_free_lives", "Five free lives instead of three",
             "More free lives raise D7 retention without hurting purchases.",
             "hold", d1_lift=0.02, d7_lift=0.015, iap_mult=0.7),
    Scenario("new_onboarding", "Shorter onboarding",
             "A shorter tutorial lifts D7 retention.",
             "investigate", drop_low_engaged_treatment=0.12),
    Scenario("harder_levels", "Harder levels 20-30",
             "Harder mid-game levels increase purchases of boosters.",
             "hold", d1_lift=-0.01, d7_lift=-0.02, iap_mult=1.1),
    Scenario("new_renderer", "New rendering engine",
             "The new renderer's smoother animation lifts D7 retention.",
             "hold", d7_lift=0.012, crash_mult=3.0),
]
BY_NAME = {s.name: s for s in SCENARIOS}


def simulate(scenario: Scenario, db_path: str | Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = scenario.n_players
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    engagement = rng.beta(2.0, 2.5, n) * 2.0                # mean about 0.9, player-level propensity
    treated = rng.random(n) < 0.5
    keep = np.ones(n, dtype=bool)
    if scenario.drop_low_engaged_treatment:
        # A bucketing bug loses some low-engaged treatment players: fewer of them, and the
        # survivors look better than they are.
        lost = treated & (engagement < 0.6) & (rng.random(n) < scenario.drop_low_engaged_treatment * 3)
        keep &= ~lost

    lift = np.zeros(DAYS)
    lift[1] = scenario.d1_lift
    lift[2:7] = np.linspace(scenario.d1_lift, scenario.d7_lift, 5)
    lift[7:] = scenario.d7_lift

    activity_rows = []
    for pid in np.flatnonzero(keep):
        e, t = engagement[pid], treated[pid]
        p_active = np.clip(BASE_ACTIVE * e + (lift if t else 0.0), 0.0, 0.97)
        p_active[0] = 1.0
        active_days = np.flatnonzero(rng.random(DAYS) < p_active)
        for day in active_days:
            sessions = 1 + rng.poisson(1.2 * e)
            levels = rng.poisson(3.0 * sessions)
            impressions = rng.poisson(3.5 * sessions * (scenario.ad_mult if t else 1.0))
            ad_rev = round(float(impressions * rng.gamma(4.0, 0.002)), 4)   # eCPM around 8 USD
            p_buy = 0.03 * e * (scenario.iap_mult if t else 1.0)
            iap = float(rng.choice(PRICE_POINTS[0], p=PRICE_POINTS[1])) if rng.random() < p_buy else 0.0
            crashes = rng.poisson(0.004 * sessions * (scenario.crash_mult if t else 1.0))
            activity_rows.append((int(pid), int(day), int(sessions), int(levels), int(impressions),
                                  ad_rev, iap, int(crashes)))

    players = [(int(pid), int(rng.integers(0, 28)),
                str(rng.choice(COUNTRIES[0], p=COUNTRIES[1])),
                "android" if rng.random() < 0.72 else "ios",
                str(rng.choice(CHANNELS[0], p=CHANNELS[1]))) for pid in np.flatnonzero(keep)]
    assignments = [(scenario.experiment_id, int(pid), "treatment" if treated[pid] else "control")
                   for pid in np.flatnonzero(keep)]

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)",
                (scenario.experiment_id, scenario.title, scenario.hypothesis, "d7_retention",
                 "iap_arpu_d7,ad_arpu_d7,crash_rate", 0.5))
    con.executemany("INSERT INTO players VALUES (?,?,?,?,?)", players)
    con.executemany("INSERT INTO assignments VALUES (?,?,?)", assignments)
    con.executemany("INSERT INTO activity VALUES (?,?,?,?,?,?,?,?)", activity_rows)
    con.commit()
    con.close()
    return db_path
