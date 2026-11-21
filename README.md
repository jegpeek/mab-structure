# mab-structure

Code and notebooks for the 3D structure of the interstellar medium traced by
the Cas A thermal light echo *Mab*, observed in seven epochs with JWST NIRCam
(F444W).  The package `mabsf` measures the 3D structure function of the
echo-illuminated dust, fits it with an ellipsoidal power law in 0.48 ly
windows, corrects for the PSF, and derives the common shape of the
substructure.

> **Status:** in preparation alongside the paper.  The input data will be
> deposited on Zenodo (DOI to be added).
>
> **Change since the first submission:** S2 is now computed with the exact
> estimator (`Config.stationary = False`); the first submission's analysis is
> `mabsf.config.FIRST_SUBMISSION`, which the notebooks use for their checks
> against the original values.  See `docs/claims_inventory.md`.

## Layout

| Path | Contents |
|---|---|
| `src/mabsf/` | the package: geometry, preprocessing, structure functions and fits, PSF, shape statistics, regions, figure |
| `notebooks/` | executed notebooks, one per step, each ending with checks against the original analysis |
| `notebooks/src/` | the notebook sources (plain Python with `# %%` cells; edit these) |
| `results/` | cached per-window fits (`window_fits/<choice>.jsonl.gz`), the PSF injection calibration, sensitivity tables |
| `validation/` | values frozen from the original analysis and `REPORT.md` |
| `data/MANIFEST.md` | the input data and where to get them |

| Notebook | Result |
|---|---|
| `00_data_and_geometry` | sky mosaics → light-echo (U, V, W) frame |
| `01_preprocessing_and_sample` | normalisation, cloud region, the 46 windows |
| `02_psf_correction` | PSF model and the injection-calibrated bias correction |
| `03_common_shape` | common shape (~13:2:1), its spread, prolateness, orientations; the S2 estimator |
| `04_regions_and_depth` | Regions 1 and 2, structure along W, trackability, drift |
| `05_science_figure` | the paper figure |
| `06_sensitivity` | every claim under every analysis choice, one at a time |

## Quick start

```bash
conda env create -f environment.yml && conda activate mabsf
# put the inputs under data/inputs/ (see data/MANIFEST.md), then
mabsf resample                       # mosaics -> UV cube (~4 min)
python scripts/run_notebooks.py      # execute all notebooks (cached fits; RECOMPUTE flags inside)
# sensitivity analysis from scratch: python scripts/resample_infills.py; python scripts/run_sensitivity.py fit
python scripts/validate.py           # summarise the checks -> validation/REPORT.md
```

## License

MIT (see `LICENSE`).
