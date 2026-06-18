"""Scanner-invariant spatial grid region aggregator.

Replaces soft K-means (RegionAggregator) with a fixed spatial grid.
Grid boundaries are defined purely by normalized [0,1] patch coordinates —
invariant to scanner, staining, and UNI embedding domain shift.

Same interface as RegionAggregator:
    forward(patch_feats, patch_coords) -> (region_feats, region_coords, assignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridRegionAggregator(nn.Module):
    """Fixed spatial grid region aggregator.

    Divides each slide into grid_h x grid_w cells using per-slide
    min-max normalized coordinates. Each non-empty cell aggregates
    its patches via shared ABMIL attention. Empty cells use a learned
    embedding to avoid zero-vectors entering downstream modules.

    Args:
        input_dim:      Patch feature dimension (default 1024 for UNI).
        grid_h:         Grid rows (default 4).
        grid_w:         Grid columns (default 4).
        region_out_dim: Output feature dimension per region (default 256).
        dropout:        Unused; kept for API compatibility.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        grid_h: int = 4,
        grid_w: int = 4,
        region_out_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_regions = grid_h * grid_w

        # Shared ABMIL attention weights applied across all grid cells
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.region_proj = nn.Sequential(
            nn.Linear(input_dim, region_out_dim),
            nn.LayerNorm(region_out_dim),
        )
        # Learnable embedding for grid cells that contain no patches
        self.empty_embed = nn.Parameter(torch.zeros(region_out_dim))
        nn.init.normal_(self.empty_embed, std=0.01)

    def forward(
        self,
        patch_feats: torch.Tensor,
        patch_coords: torch.Tensor,
    ):
        """Aggregate patches into scanner-invariant spatial grid regions.

        Args:
            patch_feats:  [B, N, input_dim]
            patch_coords: [B, N, 2]

        Returns:
            region_feats:  [B, K, region_out_dim]
            region_coords: [B, K, 2]   attention-weighted centroid per cell
            assignment:    [B, N, K]   one-hot hard assignment
        """
        B, N, D = patch_feats.shape
        K = self.num_regions
        device = patch_feats.device

        # Normalize coordinates to [0, 1] per slide (scanner-invariant)
        cmin = patch_coords.min(dim=1, keepdim=True).values    # [B, 1, 2]
        cmax = patch_coords.max(dim=1, keepdim=True).values    # [B, 1, 2]
        crange = (cmax - cmin).clamp_min(1e-6)
        coords_norm = (patch_coords - cmin) / crange           # [B, N, 2]

        # Assign each patch to a grid cell
        cx = (coords_norm[:, :, 0] * self.grid_w).long().clamp(0, self.grid_w - 1)
        cy = (coords_norm[:, :, 1] * self.grid_h).long().clamp(0, self.grid_h - 1)
        cell_idx = cy * self.grid_w + cx                       # [B, N]

        # Hard one-hot assignment: [B, N, K]
        assignment = F.one_hot(cell_idx, num_classes=K).float()

        # Shared attention scores: [B, N, 1]
        attn_scores = self.attention(patch_feats)

        # Masked softmax: for each cell k, softmax over its assigned patches only
        masked = attn_scores.expand(-1, -1, K).masked_fill(assignment == 0, -1e4)
        attn_weights = F.softmax(masked, dim=1)                # [B, N, K]

        # Weighted pool then project: [B, K, region_out_dim]
        pooled = torch.einsum("bnd,bnk->bkd", patch_feats, attn_weights)
        region_feats = self.region_proj(pooled)

        # Replace empty cells with learned embedding
        has_patches = assignment.sum(dim=1) > 0                # [B, K]
        empty = self.empty_embed.view(1, 1, -1).expand(B, K, -1)
        region_feats = torch.where(has_patches.unsqueeze(-1), region_feats, empty)

        # Region centroid coords (attention-weighted mean of patch coords)
        region_coords = torch.einsum("bnc,bnk->bkc", patch_coords, attn_weights)

        # Empty cells: use grid cell centre in original coord space
        k_idx = torch.arange(K, device=device)
        kx_c = ((k_idx % self.grid_w).float() + 0.5) / self.grid_w   # [K]
        ky_c = ((k_idx // self.grid_w).float() + 0.5) / self.grid_h  # [K]
        grid_centers = torch.stack([kx_c, ky_c], dim=1).unsqueeze(0) * crange + cmin
        region_coords = torch.where(has_patches.unsqueeze(-1), region_coords, grid_centers)

        return region_feats, region_coords, assignment
