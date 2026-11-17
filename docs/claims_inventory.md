# Claims inventory: structure-function / shape section

Every figure panel and every number quoted in the section, with the value the
original analysis gives and where the new code reproduces it. The frozen values
are in `validation/targets.json`, `targets_windows.csv`, `targets_c6.csv` and
`targets_arrays.npz`, written by `validation/extract_targets.py` from the
original (casa-fork) pipeline. Most claims are compared with `validate.py`; the
drift claim D1 and the derived trackability numbers have no automated
comparison yet (see the status note at the end of the table).

**Two calibration settings:**
- **`psf_calibration = "published"`** uses the PSF bias table as originally run.
  It contains duplicated synthetic fields: 1368 rows, but only 78 distinct
  fields.
- **`"dedup"`** removes the duplicates, and is the default in `mabsf`.

The differences are small: f12 0.437 → 0.429, common a2/a1 0.224 → 0.226,
prolateness 7.55σ → 7.59σ. The same three windows collapse either way. The
table below gives the published values, with dedup values where they differ in
the quoted precision.

| ID | Claim (text) | Quantity | Published value | Notebook |
|---|---|---|---|---|
| S0 | Analysis sample | contiguous cloud region, tiled into 0.48 ly (150 px) windows | 46 windows | 01 |
| C1 | Region 1 in-plane S2 strongly anisotropic | max/min over 9 wedges, 0.02 ≤ r ≤ 0.03 ly | 4.8–7.5× | 04 |
| C2 | Region 2 nearly isotropic in plane | same | 1.17–1.19× | 04 |
| C3 | Region 1 W variation ≈ largest in-plane variation | median W point / max-wedge S2 at the same separation | 0.92 (91% within 0.5–1.5×) | 04 |
| C4 | Region 2 (tracked) varies less along W than in plane | median and minimum W point / mean-wedge S2 | 0.61, 0.39 | 04 |
| C5 | Region 1 cannot be traced between epochs 2–4; Region 2 can | mean epoch 2–4 correlation of 0.01–0.1 ly structure, best shift | R1 0.32; R2 box 0.86 | 04 |
| C6 | Ease of following structure tracks 3D orientation, not sky-plane isotropy | Spearman ρ over the 46 windows: trackability vs inclination / anisotropy / width | −0.76 / −0.07 / +0.45 | 04 |
| D1 | Structure drifts consistently in one direction | windows whose tracked drift has dU/dW < 0 | 45 / 46 | 04 |
| D2 | Region 2 tracking velocity | (dU/dW, dV/dW) from coherent early epochs | (−0.634, +0.177) | 04 |
| C7 | Common shape about 9:2:1 (8–9:2:1) | variance-ratio ML converted at the median α (0.50): a2/a1, a3/a2 | 0.191, 0.565 | 03 |
| C7b | (direct log fit, for reference) | common a2/a1, a3/a2 | 0.224, 0.570 | 03 |
| C7c | One shape reproduces the sky-plane anisotropy | ρ(inclination, b2/b1) | −0.59 | 03 |
| C8 | Wide range of orientations | median inclination of a1 from W; KS p against isotropic orientations | 65.7°, p = 0.006 | 03 |
| C10 | Isotropy excluded | windows inconsistent with f12 = f23 = 1 at 2σ | 44 / 46 | 03 |
| C11 | Triaxial, leaning prolate | windows with f12 < f23 (equivalently a3/a2 > a2/a1); prolateness significance | 43 / 46; 7.5σ | 03 |
| C12 | Shape varies little between regions | variance-ratio means and intrinsic scatter: f12, σ(f12); f23, σ(f23) | 0.437, 0.086; 0.752, 0.052 | 03 |
| C12b | The same, in shape language | spread in a2/a1, a3/a2 | factor 1.5–1.7 (< ~2 at 95%); 15–20% (< ~25%) | 03 |
| C13 | Three windows have an unbounded long axis | collapsed fits (ratio < 0.02); 2σ upper limits on a2/a1 | 3: R1 < 0.12, (2014, 1942) < 0.61, (1864, 892) < 1 | 03 |
| P1 | PSF bias correction | additive bias in the r < 0.1 ly band as a function of tilt, from injection | `bias_by_tilt` table | 02 |

**Figure** (`science_fig_sf_shape`, 7.25 in wide, notebook 05):
- **A.** RGB composite of epochs 2/3/4 over the cloud, with all 46 windows: solid outlines for the bounded ones and dashed for the 3 upper limits. R1 and R2 are marked, with the R2 drift arrow.
- **B, D.** Zoomed RGB of Region 1 (fixed window) and Region 2 (co-moving window).
- **C, E.** PSF-deconvolved 1D S2 in 9 wedges, with the S2 points along W; angle key in C.
- **F.** a3/a2 against a2/a1 for 43 windows, plus 3 upper-limit arrows and the common-shape diamond.

**Status of the validation.** The drift claim D1 is still to be ported and
validated. The ratios in C3 and C4 and the trackability values in C5 and C6
come from the frozen arrays and the C6 table but are not yet compared
automatically.
