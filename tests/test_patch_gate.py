import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import torch
from patch_gate import PatchGate


class TestPatchGate(unittest.TestCase):
    def test_shape_preservation(self):
        torch.manual_seed(0)
        gate = PatchGate()
        feats = torch.randn(1, 500, 64)
        logits = torch.randn(1, 500)
        out = gate(feats, logits)
        self.assertEqual(out.shape, feats.shape)

    def test_identity_at_init_with_random_logits(self):
        torch.manual_seed(0)
        gate = PatchGate()
        feats = torch.randn(1, 1000, 64)
        logits = torch.randn(1, 1000)
        out = gate(feats, logits)
        ratio = (out.abs().mean() / feats.abs().mean()).item()
        self.assertGreater(ratio, 0.85, f"Output magnitude {ratio:.3f} too small at init")
        self.assertLess(ratio, 1.15, f"Output magnitude {ratio:.3f} too large at init")

    def test_bag_size_invariance(self):
        torch.manual_seed(0)
        gate = PatchGate()
        torch.manual_seed(42)
        small = torch.randn(1, 500, 64)
        small_logits = torch.randn(1, 500)
        big = small.repeat(1, 4, 1)
        big_logits = small_logits.repeat(1, 4)
        small_out = gate(small, small_logits)
        big_out = gate(big, big_logits)
        torch.testing.assert_close(small_out, big_out[:, :500, :], rtol=1e-4, atol=1e-5)

    def test_gate_amplifies_high_attention(self):
        torch.manual_seed(0)
        gate = PatchGate()
        feats = torch.ones(1, 100, 1)
        logits = torch.zeros(1, 100)
        logits[0, 0] = 10.0
        logits[0, 1] = -10.0
        out = gate(feats, logits)
        self.assertGreater(out[0, 0, 0].item(), 1.0, "high-attention patch should be amplified")
        self.assertLess(out[0, 1, 0].item(), 1.0, "low-attention patch should be down-weighted")

    def test_gradient_flows_to_tau(self):
        gate = PatchGate()
        feats = torch.randn(1, 50, 8, requires_grad=False)
        logits = torch.randn(1, 50, requires_grad=True)
        out = gate(feats, logits)
        out.sum().backward()
        self.assertIsNotNone(gate.tau.grad)
        self.assertFalse(torch.isnan(gate.tau.grad).any())


if __name__ == "__main__":
    unittest.main()

