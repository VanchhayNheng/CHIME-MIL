"""Sparse-adaptive region graph for CHIME-MIL.

Drop-in replacement for HierarchicalGraphReasoning. Replaces the fixed
top-6 spatial kNN binary mask with a learnable sparse top-k soft adjacency
built from cosine similarity over fc_node embeddings plus a Gaussian
spatial prior on region centroids. The sparsity pattern is data-driven
per slide; density is comparable to the baseline kNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphReasoning(nn.Module):
    """Sparse-adaptive region graph head.

    Args:
        feature_dim: Region feature dimension (matches RegionAggregator output).
        hidden_dim:  Hidden dim of the GRU update / node projection.
        k:           Top-k sparsity (each region attends to its k best neighbours).
        use_spatial_prior: If False, the spatial-prior term is fixed at 0
            (feature-only graph; used for ablation row 4).
        dropout:     Edge-attention dropout, matching the baseline module.
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        k: int = 10,
        use_spatial_prior: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.k = k
        self.use_spatial_prior = use_spatial_prior
        self.dropout = dropout

        self.fc_node = nn.Linear(feature_dim, hidden_dim)
        self.fc_neighbor = nn.Linear(feature_dim, hidden_dim)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.update_gate = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        if feature_dim == hidden_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(feature_dim, hidden_dim)
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def _edge_logits(
        self,
        h_nodes: torch.Tensor,
        region_coords: torch.Tensor,
    ) -> torch.Tensor:
        h_norm = F.normalize(h_nodes, dim=-1)
        cos_sim = torch.bmm(h_norm, h_norm.transpose(1, 2))

        if self.use_spatial_prior:
            cmin = region_coords.min(dim=1, keepdim=True).values
            cmax = region_coords.max(dim=1, keepdim=True).values
            coords_norm = (region_coords - cmin) / (cmax - cmin).clamp_min(1e-6)
            dist_sq = torch.cdist(coords_norm, coords_norm).pow(2)
            sigma2 = (self.log_sigma.exp() ** 2).clamp_min(1e-6)
            spatial = torch.exp(-dist_sq / (2.0 * sigma2))
        else:
            spatial = torch.zeros_like(cos_sim)

        return self.alpha * cos_sim + self.beta * spatial

    def _sparse_topk_softmax(self, edge_logits: torch.Tensor) -> torch.Tensor:
        K = edge_logits.shape[-1]
        k = min(self.k, K)
        _, topk_idx = torch.topk(edge_logits, k=k, dim=-1)
        mask = torch.zeros_like(edge_logits, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        neg_inf = torch.finfo(edge_logits.dtype).min
        masked_logits = edge_logits.masked_fill(~mask, neg_inf)
        return F.softmax(masked_logits, dim=-1)

    def forward(
        self,
        region_feats: torch.Tensor,
        region_coords: torch.Tensor,
        return_A: bool = False,
    ):
        h_nodes = self.fc_node(region_feats)
        h_neighbors = self.fc_neighbor(region_feats)

        edge_logits = self._edge_logits(h_nodes, region_coords)
        A = self._sparse_topk_softmax(edge_logits)
        A = F.dropout(A, p=self.dropout, training=self.training)
        neighbor_messages = torch.bmm(A, h_neighbors)
        b, k_dim, _ = h_nodes.shape
        h_updated = self.update_gate(
            neighbor_messages.reshape(b * k_dim, -1),
            h_nodes.reshape(b * k_dim, -1),
        ).view(b, k_dim, -1)
        h_updated = self.norm(h_updated + self.residual(region_feats))

        global_scores = self.global_attention(h_updated)
        region_importance = torch.softmax(global_scores, dim=1)
        slide_embedding = (h_updated * region_importance).sum(dim=1)
        region_importance = region_importance.squeeze(-1)

        if return_A:
            return slide_embedding, region_importance, h_updated, A
        return slide_embedding, region_importance, h_updated
