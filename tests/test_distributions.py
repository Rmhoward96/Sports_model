import pytest

from sportsmodel.model import distributions as d


def _vec(p1b=0.0, p2b=0.0, p3b=0.0, phr=0.0, pbb=0.0, pk=0.0, pout=1.0):
    return {
        "p_1b": p1b, "p_2b": p2b, "p_3b": p3b, "p_hr": phr,
        "p_bb": pbb, "p_k": pk, "p_out": pout,
    }


def test_poisson_binomial_pmf_sums_to_one():
    pmf = d.poisson_binomial_pmf([0.2, 0.3, 0.25, 0.1])
    assert sum(pmf) == pytest.approx(1.0)
    assert len(pmf) == 5  # 0..4 successes


def test_hits_over_line_matches_hand_calc():
    # Two PA, each a 50% hit chance. P(0 hits)=0.25 -> P(over 0.5)=0.75.
    vecs = [_vec(p1b=0.5, pout=0.5), _vec(p1b=0.5, pout=0.5)]
    pmf = d.hits_pmf(vecs)
    assert d.prob_over(pmf, 0.5) == pytest.approx(0.75)
    assert d.prob_over(pmf, 1.5) == pytest.approx(0.25)  # P(2 hits) = 0.25


def test_total_bases_distribution():
    # One PA that is a HR w.p. 0.5 else out: TB is 4 w.p. 0.5, else 0.
    pmf = d.total_bases_pmf([_vec(phr=0.5, pout=0.5)])
    assert pmf[0] == pytest.approx(0.5)
    assert pmf[4] == pytest.approx(0.5)
    assert d.prob_over(pmf, 1.5) == pytest.approx(0.5)


def test_at_least_one_hr():
    # Three PA at 10% HR each: 1 - 0.9^3.
    assert d.prob_at_least_one_hr([0.1, 0.1, 0.1]) == pytest.approx(1 - 0.9**3)


def test_prob_over_dist_pmf_at_book_line():
    # Evaluate the model's own pmf at whatever line the book posts, not a default.
    dist = {"kind": "pmf", "pmf": [0.30, 0.40, 0.20, 0.08, 0.02]}
    assert d.prob_over_dist(dist, 0.5) == pytest.approx(0.70)   # P(>=1)
    assert d.prob_over_dist(dist, 1.5) == pytest.approx(0.30)   # P(>=2)
    assert d.prob_over_dist(dist, 2.5) == pytest.approx(0.10)   # P(>=3)


def test_prob_over_dist_normal():
    # Normal(mean=17, sd=3): P(> 17.5) just under half.
    dist = {"kind": "normal", "mean": 17.0, "sd": 3.0}
    assert d.prob_over_dist(dist, 17.5) == pytest.approx(0.434, abs=0.01)


def test_prob_over_dist_empty_is_nan():
    p = d.prob_over_dist(None, 0.5)
    assert p != p  # NaN
