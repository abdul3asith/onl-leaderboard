"""The certification math: the Hoeffding-Bentkus bound and the Wilson interval."""

import json
import math
import random
from pathlib import Path

import pytest

from bench.certify import (
    ALPHA,
    DELTA,
    binom_cdf,
    certify,
    hb_pvalue,
    hb_upper,
    wilson_interval,
)


def test_rule_constants():
    assert ALPHA == 0.05 and DELTA == 0.05


def test_binom_cdf_against_exact_sums():
    assert binom_cdf(0, 10, 0.3) == pytest.approx(0.7**10)
    assert binom_cdf(1, 10, 0.3) == pytest.approx(0.7**10 + 10 * 0.3 * 0.7**9)
    exact = sum(math.comb(100, i) * 0.05**i * 0.95 ** (100 - i) for i in range(4))
    assert binom_cdf(3, 100, 0.05) == pytest.approx(exact)
    assert binom_cdf(-1, 10, 0.3) == 0.0 and binom_cdf(10, 10, 0.3) == 1.0
    assert binom_cdf(3, 10, 0.0) == 1.0 and binom_cdf(3, 10, 1.0) == 0.0
    assert 0.0 < binom_cdf(5000, 100_000, 0.05) < 1.0  # large n neither overflows nor vanishes


def test_pvalue_is_the_smaller_of_hoeffding_and_bentkus():
    # One failure in 100, tested against a 5% rate, worked by hand from the two formulas.
    kl = 0.01 * math.log(0.01 / 0.05) + 0.99 * math.log(0.99 / 0.95)
    hoeffding = math.exp(-100 * kl)
    bentkus = math.e * (0.95**100 + 100 * 0.05 * 0.95**99)
    assert hoeffding < bentkus
    assert hb_pvalue(1, 100, 0.05) == pytest.approx(hoeffding)
    # Bentkus is the smaller one when the observed rate sits just under the tested rate.
    n, count, risk = 1000, 40, 0.05
    kl = 0.04 * math.log(0.04 / risk) + 0.96 * math.log(0.96 / (1 - risk))
    assert math.e * binom_cdf(count, n, risk) < math.exp(-n * kl)
    assert hb_pvalue(count, n, risk) == pytest.approx(math.e * binom_cdf(count, n, risk))


def test_pvalue_is_one_at_or_below_the_observed_rate_and_falls_above_it():
    assert hb_pvalue(10, 100, 0.10) == 1.0
    assert hb_pvalue(10, 100, 0.05) == 1.0
    values = [hb_pvalue(10, 100, r / 100) for r in range(10, 101)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    assert values[-1] == 0.0


def test_upper_with_no_failures_has_a_closed_form():
    # With zero failures Hoeffding's term is (1 - r)^n, so the bound solves (1 - r)^n = delta.
    for n in (10, 100, 1000):
        assert hb_upper(0, n) == pytest.approx(1 - DELTA ** (1 / n), abs=1e-12)


def test_upper_is_where_the_pvalue_crosses_delta():
    for count, n in ((0, 100), (1, 100), (7, 300), (50, 100)):
        upper = hb_upper(count, n)
        assert hb_pvalue(count, n, upper) <= DELTA
        assert hb_pvalue(count, n, upper - 1e-9) > DELTA


def test_upper_bounds():
    assert hb_upper(100, 100) == 1.0
    for count in range(101):
        assert count / 100 <= hb_upper(count, 100) <= 1.0


def test_upper_grows_with_failures_and_shrinks_with_episodes_and_delta():
    uppers = [hb_upper(count, 100) for count in range(101)]
    assert all(a < b for a, b in zip(uppers, uppers[1:], strict=False))
    assert hb_upper(5, 100) > hb_upper(10, 200) > hb_upper(50, 1000)  # same 5% observed
    assert hb_upper(3, 100, delta=0.01) > hb_upper(3, 100, delta=0.05) > hb_upper(3, 100, 0.2)


def _clopper_pearson_upper(count: int, n: int, delta: float) -> float:
    """The exact one-sided binomial bound. Any valid bound must be at least this loose."""
    low, high = count / n, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if binom_cdf(count, n, mid) > delta else (low, mid)
    return high


@pytest.mark.parametrize("n", [20, 100, 500])
def test_upper_is_never_tighter_than_the_exact_binomial_bound(n):
    for count in range(0, n + 1, max(1, n // 20)):
        assert hb_upper(count, n) >= _clopper_pearson_upper(count, n, DELTA) - 1e-12


@pytest.mark.parametrize("rate", [0.01, 0.05, 0.2])
def test_upper_covers_the_true_rate_at_least_95_percent_of_the_time(rate):
    rng = random.Random(1000)
    n, trials = 100, 2000
    uppers = {count: hb_upper(count, n) for count in range(n + 1)}
    misses = sum(uppers[sum(rng.random() < rate for _ in range(n))] < rate for _ in range(trials))
    assert misses / trials <= DELTA


def test_certified_exactly_when_the_pvalue_at_alpha_is_at_most_delta():
    for n in (100, 300):
        for count in range(n + 1):
            assert certify(count, n)["certified"] == (hb_pvalue(count, n, ALPHA) <= DELTA)


def test_certify_fields_at_100_episodes():
    clean = certify(0, 100)
    assert clean == {
        "aif_count": 0,
        "aif_rate": 0.0,
        "aif_upper": pytest.approx(1 - 0.05**0.01),
        "certified": True,
    }
    one = certify(1, 100)
    assert one["aif_rate"] == 0.01 and one["aif_upper"] > ALPHA and one["certified"] is False


def test_bad_inputs_are_refused():
    for count, n in ((-1, 10), (11, 10), (0, 0)):
        with pytest.raises(ValueError):
            hb_upper(count, n)
    with pytest.raises(ValueError):
        hb_pvalue(1, 10, 1.5)
    for delta in (0.0, 1.0):
        with pytest.raises(ValueError):
            hb_upper(1, 10, delta)


def test_wilson_matches_lerobot():
    low, high = wilson_interval(9, 10)  # the example in lerobot's own docstring
    assert (round(low, 3), round(high, 3)) == (0.596, 0.982)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0


def test_wilson_matches_the_intervals_in_the_committed_rows():
    rows = [json.loads(p.read_text()) for p in Path("results").glob("*.json")]
    evaluated = [r for r in rows if r["episodes"]]
    assert evaluated
    for row in evaluated:
        successes = round(row["success"] * row["episodes"] / 100)
        low, high = wilson_interval(successes, row["episodes"])
        assert (100 * low, 100 * high) == pytest.approx((row["ci_low"], row["ci_high"]))
