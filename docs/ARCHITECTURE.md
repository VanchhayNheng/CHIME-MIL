# CHIME_MIL

## Goal

Turn the original CHIME_MIL hierarchy into a single end-to-end model where patch-, region-, and graph-level reasoning all contribute directly to the final prediction.

## Main changes from v1

- Keep the patch -> region -> graph hierarchy.
- Remove stage-wise transfer as the main mechanism.
- Add deep supervision at patch and region levels.
- Add learned fusion across patch, region, and graph slide embeddings.
- Keep causal intervention as an auxiliary regularizer on region importance.

## Forward path

1. Patch branch:
   - Attention MIL over patch features
   - Produces `z_patch` and auxiliary `y_patch`

2. Region branch:
   - Partition each slide into a fixed 4x4 spatial grid (16 cells) defined on
     per-slide min-max normalized patch coordinates -- invariant to scanner,
     staining, and feature-embedding domain shift by construction. There is no
     learned or feature-based clustering (no FPS, no soft K-means).
   - Within each non-empty cell, pool patches with a shared ABMIL attention
     head; empty cells fall back to a learned embedding.
   - Region attention pooling over the 16 cell embeddings produces `z_region`
     and auxiliary `y_region`

3. Graph branch:
   - Graph reasoning over region embeddings
   - Produces `z_graph`, region importance, and auxiliary `y_graph`

4. Fusion head:
   - Learn weights over `z_patch`, `z_region`, `z_graph`
   - Produce final logits `y_final`

## Training loss

`L_total = L_final + lambda_patch * L_patch + lambda_region * L_region + lambda_graph * L_graph + lambda_causal * L_causal`

## Design principles

- One model, one forward pass
- Every level is used at inference time
- Keep graph shallow for LOSO robustness
- Use causal masking as support, not as the whole architecture
