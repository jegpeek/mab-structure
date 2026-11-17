# Input data

The analysis starts from the star- and galaxy-removed JWST NIRCam F444W
mosaics and the 3D position of the scattering surface. These are about 5 GB in
total, too large for the repository, and are deposited on Zenodo
(DOI: *to be added at release*).

Download them into `data/inputs/`, or point `$MABSF_DATA` at a directory that
contains `inputs/`. Then check that everything is in place:

```python
import mabsf.data
mabsf.data.check_inputs()
```

| Path under `inputs/` | Size | What it is |
|---|---|---|
| `mosaics/mosaic_<visit>_F444W_modB_pixscale60_infill_v13.fits` (7 files) | 7 × 659 MB | One file per epoch: 60 mas mosaic, 3853 × 3802 px, with stars and galaxies replaced by infill. Extensions `INFILL_DRAW_1`–`3` are three independent infill realisations and `INFILL_MEAN` is their mean. The analysis uses `INFILL_DRAW_1`; the others are used only in the sensitivity analysis. |
| `pos3d/pos3d_60mas_2024.npy`, `pos3d_60mas_2025.npy` | 2 × 176 MB | (x, y, z) [ly] of each mosaic pixel on the light-echo paraboloid, in the Cas A-centred frame, for two reference dates. Positions at each epoch are interpolated linearly in time. |
| `psf/PSF_NIRCam_in_flight_opd_filter_F444W.fits` | 7 MB | WebbPSF in-flight F444W model (OVERDIST extension). |
| `psf/F444W_60583_b_60_i2d.fits` | 144 MB | One epoch's mosaic with stars kept, used to stack stars and check the PSF model. |

The visits are 05451001001, 05451002001, 05451012001, 05451003001,
05451004001, 08527029001 and 05451005001 (time order; see
`mabsf.config.EPOCHS`).

**Provenance.**
- **Mosaics.** Built from the MAST level-3 products by the team's star and galaxy removal and infilling pipeline (described in the paper's methods).
- **3D positions.** These come from the light-echo geometry: the paraboloid for the light-travel time of each pixel, at the adopted Cas A distance. They were mapped onto the 60 mas grid by world coordinates; `mabsf.geometry` documents the frame.

`scripts/prepare_zenodo.py` collects these files and writes their checksums
to `data/zenodo_checksums.txt`. Once the deposit is public, `mabsf.data.fetch()`
uses those checksums to verify the download.

## Products (made by the package, under `data/products/`)

| File | Made by | Size |
|---|---|---|
| `uv_cube_draw1.npy`, `U_grid.npy`, `V_grid.npy`, `epoch_w.npy`, `backgrounds.npy` | `mabsf resample` (notebook 00) | 430 MB + 2 × 61 MB |
| `edge_mask_r50.npy` | first window read | 108 MB |
| `band_e<k>.npy` (7) | `mabsf.preprocess.band_maps` (notebook 01) | 7 × 61 MB |
| `psf_f444w_uv.npz` | `mabsf.psf.build_psf_inputs` (notebook 02) | 3 MB |
