# Results

5-fold LOSO tumor-vs-normal across five hospitals (5,036 WSIs).
Features: GenBio-PathFM 4608-d CLS at 20x.

> **Evaluation regime.** The headline, ablation chain, and reading below
> are the **deterministic single-seed re-run** (`PYTHONHASHSEED=42`,
> `CUBLAS_WORKSPACE_CONFIG=:4096:8`, md5 hospital-stratified val split),
> reproducing exactly from the committed `results/rerun/<row>/fold_*/result.json`.
> These supersede the earlier 3-seed numbers. The baseline bootstrap below
> is **also deterministic** (recomputed 2026-06-02 against the 0.8434 / 0.8174
> headline). The remaining sections (score-scale heterogeneity, what did not
> work) are **prior-regime analyses** and are flagged where they appear.

## Headline

Per-slide flat fusion of two single-scale `mc_sa` LOSO tracks
(`fused_prob = 0.5·(prob_256 + prob_392)`):

| Method                            | Mean AUC | Notes |
|-----------------------------------|---------:|-------|
| UNI Top-4 ensemble (prior)        | 0.8346   | single-seed |
| **GenBio mc_sa @ (256+392) flat-fuse** | **0.8434** | deterministic 5-fold LOSO per-fold-mean |

Per-fold breakdown (verified, reproduces from committed `result.json`):

| Fold | Hospital   | AUC₂₅₆ | AUC₃₉₂ | AUC_fus | ΔAUC |
|-----:|------------|-------:|-------:|--------:|-----:|
| 1 | Site A  | 0.7893 | 0.7790 | 0.7929 | +0.0036 |
| 2 | Site B    | 0.8620 | 0.8657 | 0.8704 | +0.0047 |
| 3 | Site C | 0.8306 | 0.8229 | 0.8354 | +0.0048 |
| 4 | Site D    | 0.8278 | 0.8274 | 0.8398 | +0.0120 |
| 5 | Site E   | 0.8609 | 0.8682 | 0.8783 | +0.0101 |
| **Mean** | | **0.8341** | **0.8326** | **0.8434** | **+0.0092** |

- Fusion gain **+0.0092** over the best single scale (256 @ 0.8341);
  per-fold ΔAUC all positive → multi-scale carries complementary signal.
- **Pooled-LOSO bootstrap** (`fusion/fused_headline_ci.py`, seed 42,
  10k resamples, N=5036): pooled fused AUC = **0.8174 [95% CI 0.8047,
  0.8295]**; fused − best-single paired on the same resamples =
  **+0.0088 [+0.0048, +0.0127], two-sided p < 0.0001 (significant)**.
- Pooled point AUC (0.8174) mixes all hospitals and is lower than the
  per-fold-mean (0.8434) — Site A (52% of slides) dominates the pool.
  Paper leads with per-fold-mean; CI / significance from pooled.

Provenance pinned in `fusion/HEADLINE_PROVENANCE.md`.

## Ablation chain (deterministic, single-scale @256 unless noted)

5-fold LOSO mean test AUC. Every row uses the same grid `CHIME_MIL`
model — the old "soft_assign" row is a hospital-stratified validation
split, **not** soft assignment (see `docs/ARCHITECTURE.md`).

| # | Variant                                   | Trainer                            | Mean AUC | Δ vs baseline |
|--:|-------------------------------------------|------------------------------------|---------:|--------------:|
| 1 | Baseline                                  | `train_loso_genbio.py`             | 0.8204   | —             |
| 2 | meancenter                                | `train_loso_meancenter_genbio.py`  | 0.8274   | +0.0070       |
| 3 | stratified-val                            | `train_loso_sa_genbio.py`          | 0.8222   | +0.0018       |
| 4 | **meancenter + stratified-val (mc_sa)**   | `train_loso_mc_sa_genbio.py`       | **0.8341** | +0.0137     |
| 5 | e09 patch-only                            | `train_loso_e09_genbio.py`         | 0.8196   | −0.0008       |
| 6 | **mc_sa @ (256+392) flat-fuse**           | `fusion/late_fusion_256_392.py`    | **0.8434** | +0.0230     |

Per-fold AUCs for each row are in `results/rerun/<row>/fold_*/result.json`.

## Reading

- **The two cross-hospital alignment levers are super-additive.**
  Mean-centering alone adds +0.0070; stratified validation alone adds
  only +0.0018. Their sum would predict 0.8204 + 0.0088 = 0.8292, yet
  combined `mc_sa` reaches **0.8341 — a +0.0049 super-additive excess**.
  Equivalently, stratified-val contributes **+0.0067 on top of
  mean-centering** versus +0.0018 in isolation: first-moment site
  alignment and a hospital-matched val split reinforce each other.
- **Mean-centering is the single largest component** (+0.0070), but
  unlike the prior 3-seed read it is **not the entire lift** — the
  combination is what wins.
- **e09 (patch-head-only, no region/graph branches) sits at 0.8196**,
  marginally below baseline, confirming the region + graph branches
  carry the model's cross-hospital generalization rather than hurting it.
- **Multi-scale 256+392 flat-fuse adds a further +0.0092** over the best
  single scale, with all five per-fold deltas positive and the pooled
  paired test significant (p < 0.0001) — the clearest, most defensible
  gain in the chain.

---

> **Mixed regime below.** The baseline significance section has been
> **recomputed against the deterministic headline** (fold-mean 0.8434,
> pooled 0.8174) using the retained per-slide baseline predictions in
> `runs/baselines/`; see `analysis/bootstrap_headline_vs_baselines.py`. The
> later sections (score-scale heterogeneity, what did not work) are still
> prior-regime analyses, flagged where they appear.

## Baseline bootstrap comparison (deterministic)

Paired **pooled-LOSO** slide-level bootstrap, B=10,000, seed=42,
Benjamini-Hochberg-adjusted, against the deterministic headline (fold-mean
0.8434, pooled 0.8174). Delta = headline_pool - baseline_pool (positive =>
headline above). All baselines on the same frozen GenBio-PathFM features;
every baseline fold-mean AUC reproduces the paper's main table exactly.

| Baseline   | fold-AUC | pool-AUC | Delta_pool | 95% CI             | p_raw  | p_BH   | sig |
|------------|---------:|---------:|-----------:|:-------------------|-------:|-------:|:---:|
| ABMIL      | 0.8272   | 0.8167   | +0.0008    | [-0.0062, +0.0080] | 0.832  | 0.832  | ns  |
| CLAM-SB    | 0.8208   | 0.8058   | +0.0117    | [+0.0044, +0.0191] | 0.0020 | 0.0033 | **  |
| CLAM-MB    | 0.8274   | 0.8068   | +0.0107    | [+0.0031, +0.0185] | 0.0048 | 0.0062 | **  |
| TransMIL   | 0.8073   | 0.7827   | +0.0347    | [+0.0249, +0.0446] | <1e-4  | <1e-4  | *** |
| DSMIL      | 0.8165   | 0.7867   | +0.0307    | [+0.0213, +0.0401] | <1e-4  | <1e-4  | *** |
| RoFormer   | 0.8125   | 0.7780   | +0.0394    | [+0.0313, +0.0479] | <1e-4  | <1e-4  | *** |
| ACMIL      | 0.8386   | 0.8236   | -0.0062    | [-0.0132, +0.0008] | 0.083  | 0.093  | ns  |
| MaxPool256 | 0.8429   | 0.8319   | -0.0145    | [-0.0223, -0.0064] | 0.0002 | 0.0005 | dd  |
| MaxPool392 | 0.8461   | 0.8296   | -0.0122    | [-0.0201, -0.0042] | 0.0022 | 0.0033 | d   |

`d`/`dd` = baseline **significantly exceeds** the headline on pooled AUC
(MaxPool392 p_BH 0.003; MaxPool256 p_BH 0.0005).

**Reading:** the headline significantly beats RoFormer/TransMIL/DSMIL
(p<0.001) and CLAM-SB/MB (p<0.01, BH); is statistically indistinguishable from
ABMIL (Delta_pool +0.0008, p=0.83); and is **significantly exceeded by both
MaxPool variants** on pooled AUC (nominally by ACMIL too, ns). Note the
fold-mean vs pooled divergence: the headline leads on fold-mean AUC (0.8434)
but sits mid-pack on pooled (0.8174) because Site A is 52% of slides -- the
score-scale-heterogeneity effect below. This **reinforces** the
foundation-model-ceiling / deployment-not-discrimination framing: MaxPool's
pooled-AUC edge over the headline is now statistically significant. Both
MaxPool variants are folded into this table (they are no longer an untested
"open gap"). Full provenance and the reconstructed script are pinned in the
paper repo's `BOOTSTRAP_DETERMINISTIC.md`.

## Cross-hospital score-scale heterogeneity (methodology note)

Slide-pooled AUC differs from LOSO fold-mean AUC because per-hospital
probability distributions are differently calibrated. With p̂_y = mean
predicted P(class=1) for slides of true class y, the **decision-margin
gap** g_h = p̂_1 − p̂_0 varies across hospitals:

|                    | gap range (5 hospitals) | gap variance |
|--------------------|:-----------------------:|-------------:|
| MaxPool392         | 0.326 – 0.371           | 0.0004       |
| distpool (meanmax) | 0.335 – 0.481           | 0.0036       |

Lower gap-variance means more consistent per-hospital calibration; the
slides land on a more coherent ROC when pooled. This is the structural
reason the deterministic headline's fold-mean (0.8434) sits well above
its pooled AUC (0.8174). Future multi-site LOSO MIL evaluations should
report both summary statistics and the per-hospital gap-variance.

## What did not work *(prior regime)*

All on frozen GenBio-PathFM 4608-d features, same LOSO protocol.

| Probe | Result | Source |
|-------|--------|--------|
| Causal loss A/B (λ_c=0 vs λ_c=0.3) | Δ = +0.0022 mean AUC, within fold-std (~0.025); **no measurable lift** | `nocausal/`, `causal_control/`, 2026-05-16 |
| Per-site CORAL whitening | All whiten configs 0.72–0.77 AUC — **strongly negative** | `distpool_dro/`, screen 2026-05-18 |
| GroupDRO objective (η sweep 0/0.1) | Neutral, no detectable effect | `distpool_dro/`, screen 2026-05-18 |
| Distribution pooling head (meanmax) | scale392 3-seed = 0.8482 fold-mean (+0.0021 vs MaxPool392) but slide-pool −0.0036, **p=0.254 (ns)** — parity not significance | `distpool_dro/results/RESULT.md`, 2026-05-19 |
| Compact prototype aggregation (softhist / VLAD-lite / VLAD-norm / fisher-lite × K × PCA) | 10-config Site A screen, **all below the 0.8021 kill bar**; best fisherlite_K16_pca64 = 0.7918 | `distpool_dro/protoagg/`, 2026-05-19 |
| Cross-hospital SupCon contrastive on slide embeddings (WBCA) | no lift over plain mc_sa | `experiments/wbca/`, supplementary |

Together these support the claim that **on a strong frozen pathology FM,
the architectural and loss-shape lever is exhausted near MaxPool
parity.** The remaining lever for substantial improvement is the FM
itself (multi-FM ensembling or fine-tuning), explicitly out of scope for
this paper.
