# mab-structure

Code and notebooks for the 3D structure of the interstellar medium traced by
the Cas A thermal light echo *Mab*, observed in seven epochs with JWST NIRCam
(F444W).  The package `mabsf` measures the 3D structure function of the
echo-illuminated dust, fits it with an ellipsoidal power law in 0.48 ly
windows, corrects for the PSF, and derives the common shape of the
substructure.

> **Status:** in preparation alongside the paper.  The sensitivity-analysis
> notebook and full documentation are still to come; the input data will be
> deposited on Zenodo (DOI to be added).

## Layout

| Path | Contents |
|---|---|
| `src/mabsf/` | the package: geometry, preprocessing, structure functions and fits, PSF, shape statistics, regions, figure |
| `notebooks/` | executed notebooks, one per step, each ending with checks against the original analysis |
| `notebooks/src/` | the notebook sources (plain Python with `# %%` cells; edit these) |
| `results/` | cached per-window fits and the PSF injection calibration |
| `validation/` | values frozen from the original analysis and `REPORT.md` |
| `data/MANIFEST.md` | the input data and where to get them |

| Notebook | Result |
|---|---|
| `00_data_and_geometry` | sky mosaics → light-echo (U, V, W) frame |
| `01_preprocessing_and_sample` | normalisation, cloud region, the 46 windows |
| `02_psf_correction` | PSF model and the injection-calibrated bias correction |
| `03_common_shape` | common shape (~9:2:1), its spread, prolateness, orientations |
| `04_regions_and_depth` | Regions 1 and 2, structure along W, trackability, drift |
| `05_science_figure` | the paper figure |

## Quick start

```bash
conda env create -f environment.yml && conda activate mabsf
# put the inputs under data/inputs/ (see data/MANIFEST.md), then
mabsf resample                       # mosaics -> UV cube (~4 min)
python scripts/run_notebooks.py      # execute all notebooks
python scripts/validate.py           # summarise the checks -> validation/REPORT.md
```

## License

MIT (see `LICENSE`).
