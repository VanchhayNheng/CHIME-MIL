import sys, unittest, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chime_mil import CHIME_MIL

class TestChimeMilAdagraph(unittest.TestCase):
    def _make(self, **kw):
        d=dict(input_dim=64,hidden_dim=32,num_classes=2,num_regions=16,dropout=0.3)
        d.update(kw); return CHIME_MIL(**d)
    def test_patch_gate_flag_inserts_module(self):
        model = self._make(use_patch_gate=True)
        from patch_gate import PatchGate
        self.assertIsInstance(model.patch_gate, PatchGate)
    def test_adaptive_graph_flag_swaps_head(self):
        model = self._make(use_adaptive_graph=True)
        from graph_reasoning_adaptive import AdaptiveGraphReasoning
        self.assertIsInstance(model.graph_head, AdaptiveGraphReasoning)
    def test_adaptive_graph_adds_A_key(self):
        torch.manual_seed(0)
        model = self._make(use_adaptive_graph=True).eval()
        x=torch.randn(1,200,64); c=torch.rand(1,200,2)
        with torch.no_grad(): out=model(x,c)
        self.assertIn("A",out)
        self.assertEqual(out["A"].shape,(1,16,16))
    def test_full_adagraph_forward_smoke(self):
        torch.manual_seed(0)
        model = self._make(use_patch_gate=True,use_adaptive_graph=True).eval()
        x=torch.randn(1,300,64); c=torch.rand(1,300,2)
        with torch.no_grad(): out=model(x,c)
        self.assertEqual(out["logits"].shape,(1,2))
        self.assertIn("A",out)
    def test_k_and_spatial_prior_passthrough(self):
        model = self._make(use_adaptive_graph=True,graph_k=8,graph_use_spatial_prior=False)
        self.assertEqual(model.graph_head.k,8)
        self.assertFalse(model.graph_head.use_spatial_prior)
    def test_K_neq_16_supported(self):
        model = self._make(num_regions=12,use_adaptive_graph=True).eval()
        x=torch.randn(1,200,64); c=torch.rand(1,200,2)
        with torch.no_grad(): out=model(x,c)
        self.assertEqual(out["A"].shape,(1,12,12))
if __name__=='__main__': unittest.main()
