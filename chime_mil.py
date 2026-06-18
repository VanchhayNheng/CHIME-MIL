import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_head import PatchMILHead
from region_aggregator_grid import GridRegionAggregator
from graph_reasoning import HierarchicalGraphReasoning
from graph_reasoning_adaptive import AdaptiveGraphReasoning
from patch_gate import PatchGate

class FFTFusionLayer(nn.Module):
    """Enrich patch features with spectral magnitude via 1D rfft."""
    def __init__(self, uni_dim=1024, fft_proj_dim=64, dropout=0.3):
        super().__init__()
        self.fft_proj = nn.Sequential(nn.Linear(uni_dim//2+1, fft_proj_dim), nn.LayerNorm(fft_proj_dim), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(uni_dim+fft_proj_dim, uni_dim), nn.LayerNorm(uni_dim), nn.GELU(), nn.Dropout(dropout))
    def forward(self, x):
        fft_mag = torch.log1p(torch.abs(torch.fft.rfft(x, dim=1)))
        fft_feat = self.fft_proj(fft_mag)
        return self.fusion(torch.cat([x, fft_feat], dim=1)) + x

class RegionMILHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_classes))
    def forward(self, region_feats):
        attn = F.softmax(self.attention(region_feats), dim=1)
        slide_emb = (region_feats * attn).sum(dim=1)
        logits = self.classifier(slide_emb)
        return slide_emb, logits, attn.squeeze(-1)

class MultiLevelFusion(nn.Module):
    def __init__(self, patch_dim=1024, region_dim=1024, graph_dim=256, fusion_dim=256, num_classes=2, dropout=0.3, equal_weight_fusion=False):
        super().__init__()
        self.equal_weight = equal_weight_fusion
        self.patch_proj = nn.Linear(patch_dim, fusion_dim)
        self.region_proj = nn.Linear(region_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        self.weight_net = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Tanh(), nn.Linear(fusion_dim, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, num_classes))
    def forward(self, z_patch, z_region, z_graph):
        projected = torch.stack([self.patch_proj(z_patch), self.region_proj(z_region), self.graph_proj(z_graph)], dim=1)
        if self.equal_weight:
            weights = projected.new_full((projected.size(0), projected.size(1), 1), 1.0 / projected.size(1))
        else:
            weights = torch.softmax(self.weight_net(projected), dim=1)
        fused = (projected * weights).sum(dim=1)
        return fused, self.classifier(fused), weights.squeeze(-1)

class CHIME_MIL(nn.Module):
    """CHIME-MIL hierarchical MIL model with optional adagraph extensions.

    Args:
        use_patch_gate: Insert PatchGate before region aggregator.
        use_adaptive_graph: Replace HierarchicalGraphReasoning with AdaptiveGraphReasoning.
        graph_k: Top-k for AdaptiveGraphReasoning.
        graph_use_spatial_prior: Spatial-prior flag for AdaptiveGraphReasoning.
    """
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=2, num_regions=16,
                 dropout=0.3, use_fft=False, input_proj_dim=None,
                 use_patch_gate=False, use_adaptive_graph=False,
                 graph_k=10, graph_use_spatial_prior=True, equal_weight_fusion=False):
        super().__init__()
        self.use_patch_gate = use_patch_gate
        self.use_adaptive_graph = use_adaptive_graph
        self.input_proj_dim = input_proj_dim
        if input_proj_dim is not None:
            self.input_proj = nn.Sequential(nn.Linear(input_dim, input_proj_dim), nn.LayerNorm(input_proj_dim), nn.GELU(), nn.Dropout(dropout))
            eff_dim = input_proj_dim
        else:
            self.input_proj = None
            eff_dim = input_dim
        self.use_fft = use_fft
        if use_fft:
            self.fft_fusion = FFTFusionLayer(uni_dim=eff_dim, fft_proj_dim=64, dropout=dropout)
        self.patch_head = PatchMILHead(input_dim=eff_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
        self.patch_gate = PatchGate() if use_patch_gate else None
        grid_h, grid_w = self._factor_grid(num_regions)
        self.region_aggregator = GridRegionAggregator(input_dim=eff_dim, grid_h=grid_h, grid_w=grid_w, region_out_dim=256)
        self.region_head = RegionMILHead(input_dim=256, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
        if use_adaptive_graph:
            self.graph_head = AdaptiveGraphReasoning(feature_dim=256, hidden_dim=hidden_dim, k=graph_k, use_spatial_prior=graph_use_spatial_prior, dropout=dropout)
        else:
            self.graph_head = HierarchicalGraphReasoning(feature_dim=256, hidden_dim=hidden_dim, k_neighbors=6, dropout=dropout)
        self.graph_classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.fusion = MultiLevelFusion(patch_dim=eff_dim, region_dim=256, graph_dim=hidden_dim, fusion_dim=hidden_dim, num_classes=num_classes, dropout=dropout, equal_weight_fusion=equal_weight_fusion)

    @staticmethod
    def _factor_grid(num_regions):
        """Return (grid_h, grid_w) for supported K values."""
        table = {12: (3, 4), 16: (4, 4), 24: (4, 6)}
        if num_regions not in table:
            raise ValueError(f"num_regions={num_regions} not supported; add to _factor_grid")
        return table[num_regions]

    def forward(self, patch_feats, patch_coords):
        """Forward pass. Returns a dict with all level outputs."""
        if self.input_proj is not None:
            patch_feats = self.input_proj(patch_feats)
        if self.use_fft:
            patch_feats = self.fft_fusion(patch_feats.squeeze(0)).unsqueeze(0)
        z_patch, y_patch, patch_importance, attn_logits = self.patch_head(patch_feats)
        feats_for_region = patch_feats
        if self.patch_gate is not None:
            feats_for_region = self.patch_gate(patch_feats, attn_logits)
        region_feats, region_coords, assignment = self.region_aggregator(feats_for_region, patch_coords)
        z_region, y_region, region_attention = self.region_head(region_feats)
        if self.use_adaptive_graph:
            z_graph, graph_importance, graph_region_feats, A = self.graph_head(region_feats, region_coords, return_A=True)
        else:
            z_graph, graph_importance, graph_region_feats = self.graph_head(region_feats, region_coords)
            A = None
        y_graph = self.graph_classifier(z_graph)
        z_fused, y_final, fusion_weights = self.fusion(z_patch, z_region, z_graph)
        out = {
            "logits": y_final, "patch_logits": y_patch, "region_logits": y_region, "graph_logits": y_graph,
            "patch_embedding": z_patch, "region_embedding": z_region, "graph_embedding": z_graph, "fused_embedding": z_fused,
            "patch_importance": patch_importance, "region_attention": region_attention,
            "graph_importance": graph_importance, "fusion_weights": fusion_weights,
            "region_feats": region_feats, "region_graph_feats": graph_region_feats,
            "region_coords": region_coords, "assignment": assignment,
        }
        if A is not None:
            out["A"] = A
        return out

    def forward_graph(self, region_feats, region_coords):
        """Re-run only the graph branch (used by causal counterfactual loss)."""
        z_graph, graph_importance, graph_region_feats = self.graph_head(region_feats, region_coords)
        y_graph = self.graph_classifier(z_graph)
        return y_graph, graph_importance, graph_region_feats
