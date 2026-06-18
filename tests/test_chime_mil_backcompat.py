import sys, unittest, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chime_mil import CHIME_MIL

class TestChimeMilBackcompat(unittest.TestCase):
    def test_defaults_construct_old_arch(self):
        model = CHIME_MIL(input_dim=64,hidden_dim=32,num_classes=2,num_regions=16,dropout=0.3)
        from graph_reasoning import HierarchicalGraphReasoning
        self.assertIsInstance(model.graph_head, HierarchicalGraphReasoning)
        self.assertTrue(model.patch_gate is None)
    def test_defaults_output_dict_keys_unchanged(self):
        torch.manual_seed(0)
        model = CHIME_MIL(input_dim=64,hidden_dim=32,num_classes=2,num_regions=16,dropout=0.3).eval()
        x=torch.randn(1,200,64); c=torch.rand(1,200,2)
        with torch.no_grad(): out=model(x,c)
        exp={"logits","patch_logits","region_logits","graph_logits","patch_embedding","region_embedding","graph_embedding","fused_embedding","patch_importance","region_attention","graph_importance","fusion_weights","region_feats","region_graph_feats","region_coords","assignment"}
        self.assertEqual(set(out.keys()),exp)
if __name__=='__main__': unittest.main()
