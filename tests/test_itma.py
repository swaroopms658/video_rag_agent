"""Unit tests for ITMA memory bank and scoring head."""

import math

import numpy as np
import pytest
import torch

from src.itma.memory_bank import MemoryBank, W_CF_MIN, W_CF_MAX
from src.itma.scoring_head import ITMAHead, FrozenScoringHead, D, P, BETA_DENSE, BETA_MEM


# ---------------------------------------------------------------------------
# MemoryBank tests
# ---------------------------------------------------------------------------

class TestMemoryBank:
    def _rand(self):
        return np.random.randn(D).astype(np.float32)

    def test_add_and_size(self):
        mb = MemoryBank(capacity=4)
        mb.add(self._rand(), self._rand(), "id1")
        mb.add(self._rand(), self._rand(), "id2")
        assert mb.size() == 2

    def test_capacity_eviction(self):
        mb = MemoryBank(capacity=3)
        for i in range(5):
            mb.add(self._rand(), self._rand(), f"id{i}")
        assert mb.size() == 3

    def test_duplicate_id_ignored(self):
        mb = MemoryBank(capacity=4)
        q, c = self._rand(), self._rand()
        mb.add(q, c, "same_id")
        mb.add(q, c, "same_id")
        assert mb.size() == 1

    def test_effective_weights_decay(self):
        mb = MemoryBank(capacity=4, lam=0.1)
        mb.add(self._rand(), self._rand(), "id0")
        mb.add(self._rand(), self._rand(), "id1")
        # After adding id1, id0's age is 1, id1's age is 0
        ws = mb.effective_weights()
        assert ws.shape == (2,)
        # id0 (age=1) has lower weight than id1 (age=0)
        assert ws[0] < ws[1]

    def test_attend_empty_returns_zero(self):
        mb = MemoryBank(capacity=4)
        q = self._rand()
        m = mb.attend(q)
        assert m.shape == (D,)
        assert np.allclose(m, 0.0)

    def test_attend_nonzero_with_entries(self):
        mb = MemoryBank(capacity=4)
        mb.add(self._rand(), self._rand(), "id0")
        mb.add(self._rand(), self._rand(), "id1")
        q = self._rand()
        m = mb.attend(q)
        assert m.shape == (D,)
        assert not np.allclose(m, 0.0)

    def test_counterfactual_positive_update(self):
        mb = MemoryBank(capacity=4, eta=0.1)
        mb.add(self._rand(), self._rand(), "id0")
        initial_wcf = list(mb._entries)[0]["w_cf"]
        mb.update_counterfactual(["id0"], [0.5], reward=1.0)
        new_wcf = list(mb._entries)[0]["w_cf"]
        assert new_wcf > initial_wcf

    def test_counterfactual_negative_update(self):
        mb = MemoryBank(capacity=4, eta=0.1)
        mb.add(self._rand(), self._rand(), "id0")
        initial_wcf = list(mb._entries)[0]["w_cf"]
        mb.update_counterfactual(["id0"], [0.5], reward=-1.0)
        new_wcf = list(mb._entries)[0]["w_cf"]
        assert new_wcf < initial_wcf

    def test_counterfactual_clipping(self):
        mb = MemoryBank(capacity=4, eta=1.0)  # large eta to force clipping
        mb.add(self._rand(), self._rand(), "id0")
        for _ in range(100):
            mb.update_counterfactual(["id0"], [1.0], reward=1.0)
        wcf = list(mb._entries)[0]["w_cf"]
        assert wcf <= W_CF_MAX

        for _ in range(100):
            mb.update_counterfactual(["id0"], [1.0], reward=-1.0)
        wcf = list(mb._entries)[0]["w_cf"]
        assert wcf >= W_CF_MIN

    def test_save_load(self, tmp_path):
        path = str(tmp_path / "mb.pkl")
        mb = MemoryBank(capacity=4, persist_path=path)
        mb.add(self._rand(), self._rand(), "id0")
        mb.add(self._rand(), self._rand(), "id1")
        mb.save()

        mb2 = MemoryBank(capacity=4, persist_path=path)
        assert mb2.size() == 2

    def test_reset(self, tmp_path):
        path = str(tmp_path / "mb.pkl")
        mb = MemoryBank(capacity=4, persist_path=path)
        mb.add(self._rand(), self._rand(), "id0")
        mb.save()
        mb.reset()
        assert mb.size() == 0


# ---------------------------------------------------------------------------
# ITMAHead tests
# ---------------------------------------------------------------------------

class TestITMAHead:
    def _make_batch(self, B=4):
        import torch.nn.functional as F
        q = F.normalize(torch.randn(B, D), dim=-1)
        c = F.normalize(torch.randn(B, D), dim=-1)
        m = torch.randn(B, D)
        cos = (q * c).sum(dim=-1)   # bounded [-1, 1] since q, c are L2-normalised
        return q, c, m, cos

    def test_output_shape(self):
        head = ITMAHead()
        q, c, m, cos = self._make_batch(4)
        out = head(q, c, m, cos)
        assert out.shape == (4,)

    def test_output_range(self):
        head = ITMAHead()
        q, c, m, cos = self._make_batch(8)
        out = head(q, c, m, cos)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_cold_start_gate(self):
        head = ITMAHead()
        # With m=zeros (empty memory) the gate should be small due to bias=-3 init
        q = torch.randn(2, D)
        c = torch.randn(2, D)
        m = torch.zeros(2, D)
        cos = torch.zeros(2)
        out = head(q, c, m, cos)
        # Gate should be near-closed: output dominated by BETA_DENSE * cos_01
        # cos_01 ≈ 0.5 when cos=0 → output ≈ BETA_DENSE * 0.5 ≈ 0.35
        # Just check it's reasonable (not 1.0 or 0.0)
        assert out.min().item() > 0.0
        assert out.max().item() < 1.0

    def test_param_count_reasonable(self):
        head = ITMAHead()
        n = head.n_params()
        # ~550K params, not 0 and not massive
        assert 400_000 < n < 2_000_000


# ---------------------------------------------------------------------------
# FrozenScoringHead tests
# ---------------------------------------------------------------------------

class TestFrozenScoringHead:
    def test_inference_no_grad(self):
        frozen = FrozenScoringHead(checkpoint_path=None)
        q = np.random.randn(D).astype(np.float32)
        c = np.random.randn(5, D).astype(np.float32)
        m = np.zeros((5, D), dtype=np.float32)
        scores = frozen.score(q, c, m)
        assert scores.shape == (5,)
        assert scores.dtype == np.float32

    def test_scores_differ_per_candidate(self):
        frozen = FrozenScoringHead(checkpoint_path=None)
        q = np.random.randn(D).astype(np.float32)
        c = np.random.randn(10, D).astype(np.float32)
        m = np.zeros((10, D), dtype=np.float32)
        scores = frozen.score(q, c, m)
        # Scores should not all be identical
        assert np.std(scores) > 1e-6

    def test_memory_changes_scores(self):
        frozen = FrozenScoringHead(checkpoint_path=None)
        q = np.random.randn(D).astype(np.float32)
        c = np.random.randn(3, D).astype(np.float32)
        m_empty = np.zeros((3, D), dtype=np.float32)
        m_full = np.random.randn(3, D).astype(np.float32)

        scores_empty = frozen.score(q, c, m_empty)
        scores_full = frozen.score(q, c, m_full)
        # Non-empty memory should change scores
        assert not np.allclose(scores_empty, scores_full)
