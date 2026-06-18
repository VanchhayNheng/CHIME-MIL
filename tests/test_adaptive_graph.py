import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import torch
from graph_reasoning_adaptive import AdaptiveGraphReasoning


class TestAdaptiveGraphReasoning(unittest.TestCase):
    def _make(self, **kw):
        defaults = dict(feature_dim=256, hidden_dim=256, k=10)
        defaults.update(kw)
        return AdaptiveGraphReasoning(**defaults)
    def test_output_contract_shapes(self):
        torch.manual_seed(0)
        mod = self._make()
        region_feats = torch.randn(2, 16, 256)
        region_coords = torch.randn(2, 16, 2)
        slide_emb, region_importance, h_updated, A = mod(region_feats, region_coords, return_A=True)
        self.assertEqual(slide_emb.shape, (2, 256))
        self.assertEqual(region_importance.shape, (2, 16))
        self.assertEqual(h_updated.shape, (2, 16, 256))
        self.assertEqual(A.shape, (2, 16, 16))

    def test_output_contract_three_value_default(self):
        """When return_A is False (default), returns 3-tuple compatible with baseline interface."""
        mod = self._make()
        rf = torch.randn(1, 16, 256)
        rc = torch.randn(1, 16, 2)
        out = mod(rf, rc)
        self.assertEqual(len(out), 3)

    def test_adjacency_is_row_stochastic_over_topk(self):
        torch.manual_seed(0)
        mod = self._make(k=10).eval()
        rf = torch.randn(1, 16, 256)
        rc = torch.randn(1, 16, 2)
        _, _, _, A = mod(rf, rc, return_A=True)
        row_sums = A.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)
        nonzero_per_row = (A > 0).sum(dim=-1)
        self.assertTrue((nonzero_per_row == 10).all().item(), f"expected exactly 10 nonzero per row, got {nonzero_per_row}")

    def test_feature_only_mode_disables_spatial(self):
        """When use_spatial_prior=False, beta is fixed at 0 and adjacency is invariant to coords."""
        torch.manual_seed(0)
        mod = self._make(use_spatial_prior=False).eval()
        rf = torch.randn(1, 16, 256)
        coords_a = torch.randn(1, 16, 2)
        coords_b = torch.randn(1, 16, 2) * 100.0
        _, _, _, A_a = mod(rf, coords_a, return_A=True)
        _, _, _, A_b = mod(rf, coords_b, return_A=True)
        torch.testing.assert_close(A_a, A_b, rtol=1e-5, atol=1e-5)

    def test_supports_non_default_K(self):
        for K, k in [(12, 8), (24, 10)]:
            with self.subTest(K=K, k=k):
                mod = self._make(k=k).eval()
                rf = torch.randn(1, K, 256)
                rc = torch.randn(1, K, 2)
                slide_emb, ri, hu, A = mod(rf, rc, return_A=True)
                self.assertEqual(slide_emb.shape, (1, 256))
                self.assertEqual(ri.shape, (1, K))
                self.assertEqual(hu.shape, (1, K, 256))
                self.assertEqual(A.shape, (1, K, K))
                self.assertTrue(((A > 0).sum(dim=-1) == k).all().item())

    def test_gradients_flow_to_alpha_beta_sigma(self):
        mod = self._make()
        rf = torch.randn(1, 16, 256)
        rc = torch.randn(1, 16, 2)
        slide_emb, _, _ = mod(rf, rc)
        slide_emb.sum().backward()
        for name in ("alpha", "beta", "log_sigma"):
            p = getattr(mod, name)
            self.assertIsNotNone(p.grad, f"{name} got no gradient")
            self.assertFalse(torch.isnan(p.grad).any(), f"{name} grad has NaN")


if __name__ == "__main__":
    unittest.main()
