import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import torch
from patch_head import PatchMILHead


class TestPatchMILHeadLogits(unittest.TestCase):
    def test_forward_returns_four_values_including_logits(self):
        torch.manual_seed(0)
        head = PatchMILHead(input_dim=64, hidden_dim=16, num_classes=2)
        x = torch.randn(1, 100, 64)
        out = head(x)
        self.assertEqual(len(out), 4, "PatchMILHead.forward must return (slide_emb, logits, attn_softmax, attn_logits)")
        slide_emb, cls_logits, attn_softmax, attn_logits = out
        self.assertEqual(slide_emb.shape, (1, 64))
        self.assertEqual(cls_logits.shape, (1, 2))
        self.assertEqual(attn_softmax.shape, (1, 100))
        self.assertEqual(attn_logits.shape, (1, 100))

    def test_attn_logits_softmax_matches_attn_softmax(self):
        torch.manual_seed(1)
        head = PatchMILHead(input_dim=64, hidden_dim=16, num_classes=2)
        x = torch.randn(1, 50, 64)
        _, _, attn_softmax, attn_logits = head(x)
        recomputed = torch.softmax(attn_logits, dim=1)
        torch.testing.assert_close(recomputed, attn_softmax, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
