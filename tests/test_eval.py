"""Sanity tests for eval_utils metric functions and analysis.stats."""

import math
import pytest

from src.eval_utils import hit_at_k, mrr, ndcg_at_k, recall_at_k, rouge_l, bleu_4
from analysis.stats import (
    paired_permutation_test,
    bootstrap_ci,
    cohens_d,
    holm_bonferroni,
    compare_systems,
)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

class TestHitAtK:
    def test_hit_in_range(self):
        assert hit_at_k(["a", "b", "c"], {"b"}, k=3) == 1

    def test_miss_in_range(self):
        assert hit_at_k(["a", "b", "c"], {"d"}, k=3) == 0

    def test_hit_outside_cutoff(self):
        # "c" is rank 3 but k=2 so it doesn't count
        assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0

    def test_hit_at_1(self):
        assert hit_at_k(["a", "b", "c"], {"a"}, k=1) == 1

    def test_empty_relevant(self):
        assert hit_at_k(["a", "b"], set(), k=2) == 0


class TestMRR:
    def test_first_rank(self):
        assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_rank(self):
        assert mrr(["a", "b", "c"], {"b"}) == pytest.approx(0.5)

    def test_third_rank(self):
        assert mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert mrr(["a", "b", "c"], {"d"}) == 0.0

    def test_multiple_relevant_takes_first(self):
        assert mrr(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)


class TestNDCGAtK:
    def test_perfect_single(self):
        assert ndcg_at_k(["a", "b"], {"a"}, k=2) == pytest.approx(1.0)

    def test_second_position(self):
        score = ndcg_at_k(["a", "b"], {"b"}, k=2)
        # DCG = 1/log2(3); IDCG = 1/log2(2)
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert score == pytest.approx(expected)

    def test_no_relevant(self):
        assert ndcg_at_k(["a", "b"], {"c"}, k=2) == 0.0

    def test_empty_retrieved(self):
        assert ndcg_at_k([], {"a"}, k=5) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_partial_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=1) == pytest.approx(0.5)

    def test_zero_recall(self):
        assert recall_at_k(["a", "b", "c"], {"d", "e"}, k=3) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set(), k=1) == 0.0


# ---------------------------------------------------------------------------
# Generation metrics (no network calls — just check return-value ranges)
# ---------------------------------------------------------------------------

class TestRougeL:
    def test_identical(self):
        assert rouge_l("hello world", "hello world") == pytest.approx(1.0)

    def test_empty_prediction(self):
        # Should return a float without crashing
        val = rouge_l("", "hello world")
        assert 0.0 <= val <= 1.0

    def test_range(self):
        val = rouge_l("the cat sat on the mat", "the cat is on the mat")
        assert 0.0 <= val <= 1.0


class TestBleu4:
    def test_identical(self):
        # sacrebleu sentence_bleu returns 100 for identical; we normalise to 1.0
        val = bleu_4("hello world today", "hello world today")
        assert val == pytest.approx(1.0, abs=0.01)

    def test_range(self):
        val = bleu_4("the quick brown fox", "a slow red dog")
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

class TestPairedPermutationTest:
    def test_identical_series_p_value(self):
        # Same series → mean diff = 0 → p ≈ 1.0
        a = [0.5, 0.6, 0.7]
        _, p = paired_permutation_test(a, a)
        assert p > 0.5

    def test_clearly_different_p_value(self):
        # Large consistent difference → small p
        a = [1.0] * 20
        b = [0.0] * 20
        _, p = paired_permutation_test(a, b)
        assert p < 0.01

    def test_returns_two_floats(self):
        diff, p = paired_permutation_test([0.5, 0.6], [0.4, 0.5])
        assert isinstance(diff, float) and isinstance(p, float)


class TestBootstrapCI:
    def test_estimate_close_to_mean(self):
        import numpy as np
        data = [0.3, 0.5, 0.7, 0.4, 0.6]
        est, lo, hi = bootstrap_ci(data)
        assert abs(est - np.mean(data)) < 1e-9

    def test_ci_contains_estimate(self):
        data = [0.3, 0.5, 0.7, 0.4, 0.6]
        est, lo, hi = bootstrap_ci(data)
        assert lo <= est <= hi

    def test_wider_with_more_variance(self):
        narrow = bootstrap_ci([0.5] * 20)
        wide = bootstrap_ci([0.0, 1.0] * 10)
        assert (wide[2] - wide[1]) >= (narrow[2] - narrow[1])


class TestCohensD:
    def test_no_difference(self):
        assert cohens_d([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_large_effect(self):
        # samples must have variance for Cohen's d to be non-zero
        a = [10.0 + i * 0.1 for i in range(10)]
        b = [0.0 + i * 0.1 for i in range(10)]
        d = cohens_d(a, b)
        assert abs(d) > 1.0

    def test_sign(self):
        a = [1.0, 1.1, 0.9]
        b = [0.0, 0.1, -0.1]
        assert cohens_d(a, b) > 0


class TestHolmBonferroni:
    def test_no_inflation_on_single(self):
        corrected = holm_bonferroni([0.03])
        assert corrected[0] == pytest.approx(0.03)

    def test_most_significant_unchanged(self):
        # The smallest p gets multiplied by n (Bonferroni-like at rank 0)
        corrected = holm_bonferroni([0.001, 0.04, 0.2])
        assert corrected[0] == pytest.approx(0.003, abs=1e-9)

    def test_monotone_in_sorted_order(self):
        p_vals = [0.01, 0.04, 0.2, 0.5]
        corrected = holm_bonferroni(p_vals)
        sorted_corrected = [corrected[i] for i in sorted(range(4), key=lambda i: p_vals[i])]
        for i in range(len(sorted_corrected) - 1):
            assert sorted_corrected[i] <= sorted_corrected[i + 1]


class TestCompareSystems:
    def test_returns_expected_keys(self):
        result = compare_systems([0.5, 0.6, 0.7], [0.4, 0.5, 0.6],
                                 system_a="A", system_b="B", n_resamples=100)
        for key in ("mean_a", "mean_b", "mean_diff", "p_value", "cohens_d", "ci95_a", "ci95_b"):
            assert key in result
