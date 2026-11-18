"""Frequentist A/B statistics, standard library only.

Sample sizes here are in the thousands per arm, so normal approximations are used for
both proportions and means (Welch's t is effectively z at these sizes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

Z95 = 1.959963984540054


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p(z: float) -> float:
    return 2.0 * (1.0 - norm_cdf(abs(z)))


@dataclass
class TestResult:
    metric: str
    control: float
    treatment: float
    abs_diff: float
    rel_diff: float          # relative to control
    ci_low: float            # 95% CI of abs_diff
    ci_high: float
    p_value: float
    n_control: int
    n_treatment: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def proportion_test(metric: str, x_c: int, n_c: int, x_t: int, n_t: int) -> TestResult:
    """Two-proportion z-test (pooled SE for the test, unpooled SE for the interval)."""
    p_c, p_t = x_c / n_c, x_t / n_t
    pooled = (x_c + x_t) / (n_c + n_t)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    se = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    diff = p_t - p_c
    z = diff / se_pooled if se_pooled else 0.0
    return TestResult(metric, p_c, p_t, diff, diff / p_c if p_c else 0.0,
                      diff - Z95 * se, diff + Z95 * se, two_sided_p(z), n_c, n_t)


def mean_test(metric: str, sum_c: float, sumsq_c: float, n_c: int,
              sum_t: float, sumsq_t: float, n_t: int) -> TestResult:
    """Welch test on means from sums and sums of squares (what SQL can aggregate)."""
    m_c, m_t = sum_c / n_c, sum_t / n_t
    var_c = max(0.0, (sumsq_c - n_c * m_c ** 2) / (n_c - 1))
    var_t = max(0.0, (sumsq_t - n_t * m_t ** 2) / (n_t - 1))
    se = math.sqrt(var_c / n_c + var_t / n_t)
    diff = m_t - m_c
    z = diff / se if se else 0.0
    return TestResult(metric, m_c, m_t, diff, diff / m_c if m_c else 0.0,
                      diff - Z95 * se, diff + Z95 * se, two_sided_p(z), n_c, n_t)


def _ratio_and_variance(y_sum, y_sq, x_sum, x_sq, xy_sum, n) -> tuple[float, float]:
    """Ratio of means R = mean(y) / mean(x) and its variance by the delta method."""
    mx, my = x_sum / n, y_sum / n
    r = my / mx
    var_y = (y_sq - n * my ** 2) / (n - 1)
    var_x = (x_sq - n * mx ** 2) / (n - 1)
    cov = (xy_sum - n * mx * my) / (n - 1)
    return r, max(0.0, (var_y - 2 * r * cov + r ** 2 * var_x) / (n * mx ** 2))


def ratio_test(metric: str, c: tuple, t: tuple) -> TestResult:
    """Compare per-unit ratios (e.g. crashes per session) where the user is the randomisation unit.

    c and t are (y_sum, y_sq, x_sum, x_sq, xy_sum, n). Treating sessions as independent would
    understate the variance, because sessions from one player are correlated.
    """
    r_c, v_c = _ratio_and_variance(*c)
    r_t, v_t = _ratio_and_variance(*t)
    se = math.sqrt(v_c + v_t)
    diff = r_t - r_c
    z = diff / se if se else 0.0
    return TestResult(metric, r_c, r_t, diff, diff / r_c if r_c else 0.0,
                      diff - Z95 * se, diff + Z95 * se, two_sided_p(z), c[5], t[5])


def srm_test(n_c: int, n_t: int, expected_treatment_share: float = 0.5) -> dict:
    """Sample ratio mismatch: chi-square goodness of fit with 1 degree of freedom."""
    total = n_c + n_t
    exp_t = total * expected_treatment_share
    exp_c = total - exp_t
    chi2 = (n_c - exp_c) ** 2 / exp_c + (n_t - exp_t) ** 2 / exp_t
    p = math.erfc(math.sqrt(chi2 / 2.0))       # survival function of chi2 with 1 df
    return {"n_control": n_c, "n_treatment": n_t, "observed_treatment_share": round(n_t / total, 4),
            "expected_treatment_share": expected_treatment_share, "chi2": round(chi2, 3),
            "p_value": p, "ok": p >= 0.001}


def mde_proportion(p: float, n_per_arm: int, power_z: float = 0.8416) -> float:
    """Smallest absolute lift detectable at alpha 0.05 (two-sided) and 80% power."""
    return (Z95 + power_z) * math.sqrt(2 * p * (1 - p) / n_per_arm)
