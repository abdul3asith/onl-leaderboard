"""The statistics behind a row's success interval and its certification.

Certification asks whether a technique's rate of acceleration-induced failures (AIF: the
native policy succeeds on an episode and the technique fails the same episode) is below a
risk budget `ALPHA`, with confidence `1 - DELTA`. The upper bound on that rate is the
Hoeffding-Bentkus bound used by Risk-Controlling Prediction Sets and Learn then Test
(Bates et al. 2021, Angelopoulos et al. 2021): it holds for any sample size, with no
normal approximation.

Pure Python on purpose: this runs locally on episode files, never on the GPU.
"""

from __future__ import annotations

import math
from statistics import NormalDist

ALPHA = 0.05  # risk budget: the largest acceptable AIF rate
DELTA = 0.05  # 1 - confidence
_Z95 = NormalDist().inv_cdf(0.975)  # two-sided 95%, derived as lerobot's Wilson interval does


def _check_counts(count: int, n: int) -> None:
    if n < 1 or not 0 <= count <= n:
        raise ValueError(f"need 0 <= count <= n and n >= 1, got count={count}, n={n}")


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= k), summed in log space so large n does not overflow."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = (
        math.exp(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
        for i in range(k + 1)
    )
    return min(1.0, math.fsum(terms))


def _h1(a: float, b: float) -> float:
    """KL divergence between Bernoulli(a) and Bernoulli(b), with 0 * log(0) = 0."""
    total = 0.0
    for x, y in ((a, b), (1.0 - a, 1.0 - b)):
        if x > 0.0:
            if y <= 0.0:
                return math.inf
            total += x * math.log(x / y)
    return total


def hb_pvalue(count: int, n: int, risk: float) -> float:
    """Hoeffding-Bentkus p-value for the null "the true rate is at least `risk`".

    `count` failures were seen in `n` episodes. Small values are evidence that the true
    rate is below `risk`. It is the smaller of Hoeffding's bound in its tight KL form and
    Bentkus's bound, e * P(Binomial(n, risk) <= count).
    """
    _check_counts(count, n)
    if not 0.0 <= risk <= 1.0:
        raise ValueError(f"risk must be in [0, 1], got {risk}")
    rate = count / n
    hoeffding = math.exp(-n * _h1(min(rate, risk), risk))
    bentkus = math.e * binom_cdf(count, n, risk)
    return min(1.0, hoeffding, bentkus)


def hb_upper(count: int, n: int, delta: float = DELTA) -> float:
    """Upper bound on the true rate that holds with probability at least 1 - delta.

    The smallest rate the p-value rejects at level `delta`. The p-value is 1 at the
    observed rate and falls as the rate rises, so bisection finds the crossing; the
    returned end is the rejected one, which keeps the bound on the safe side.
    """
    _check_counts(count, n)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    low, high = count / n, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if hb_pvalue(count, n, mid) > delta:
            low = mid
        else:
            high = mid
    return high


def certify(count: int, n: int, alpha: float = ALPHA, delta: float = DELTA) -> dict:
    """The four certification fields of a row, from `count` AIFs in `n` paired episodes."""
    upper = hb_upper(count, n, delta)
    return {
        "aif_count": count,
        "aif_rate": count / n,
        "aif_upper": upper,
        "certified": upper <= alpha,
    }


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for a success rate, as fractions. Same formula as lerobot's eval."""
    _check_counts(successes, n)
    p = successes / n
    z2 = _Z95 * _Z95
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    half = _Z95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)
