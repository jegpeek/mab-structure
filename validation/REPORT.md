# Validation report

Every result of the notebooks compared with the original analysis (values frozen by `validation/extract_targets.py`).

**122 checks passed, 0 failed; 0 notebook errors.**

| Notebook | Passed | Failed |
|---|---|---|
| 00_data_and_geometry.ipynb | 6 | 0 |
| 01_preprocessing_and_sample.ipynb | 4 | 0 |
| 02_psf_correction.ipynb | 17 | 0 |
| 03_common_shape.ipynb | 44 | 0 |
| 04_regions_and_depth.ipynb | 27 | 0 |
| 05_science_figure.ipynb | 11 | 0 |
| 06_sensitivity.ipynb | 13 | 0 |

## 00_data_and_geometry.ipynb

| | Check | Result |
|---|---|---|
| ✓ | UV cube shape | [   7. 4043. 3800.]  (expected [   7. 4043. 3800.]) |
| ✓ | U, V grid origin [ly] | [101.25773  49.86423]  (expected [101.25773  49.86423]) |
| ✓ | epoch depths W_k [ly] | [201.81771 201.86523 201.88869 201.91235 202.03328 202.33925 202.36495]  (expected [201.81771 201.86523 201.88869 201.91235 202.03328 202.33925 202.36495]) |
| ✓ | epoch sky levels | [0.38534 0.3797  0.3804  0.38158 0.38296 0.38852 0.38463]  (expected [0.38534 0.3797  0.3804  0.38158 0.38296 0.38852 0.38463]) |
| ✓ | mean flux per epoch | [0.40428 0.40355 0.40401 0.40446 0.40615 0.42504 0.42368]  (expected [0.40428 0.40355 0.40401 0.40446 0.40615 0.42504 0.42368]) |
| ✓ | finite pixels per epoch | [5388433. 4717991. 4854634. 5438358. 5465963. 5496404. 5326514.]  (expected [5388433. 4717991. 4854634. 5438358. 5465963. 5496404. 5326514.]) |


## 01_preprocessing_and_sample.ipynb

| | Check | Result |
|---|---|---|
| ✓ | number of windows | 46  (expected 46) |
| ✓ | window corners (row, col) | [92 values, max /diff/ 0]  (expected ) |
| ✓ | cloud region size [px] | 1.41813e+06  (expected 1.41813e+06) |
| ✓ | sky noise per epoch | [0.00358 0.00364 0.00364 0.00366 0.00386 0.00243 0.00368]  (expected [0.00358 0.00364 0.00364 0.00366 0.00386 0.00243 0.00368]) |


## 02_psf_correction.ipynb

| | Check | Result |
|---|---|---|
| ✓ | PSF input psf_os (sha1 match) | 1  (expected 1) |
| ✓ | PSF input star_stack (sha1 match) | 1  (expected 1) |
| ✓ | cloud Jacobian J | [[-3.66021e-03 -8.73791e-05] |
| ✓ | number of stars stacked | 52  (expected 52) |
| ✓ | published bias(tilt) a2a1 | [0.01294 0.01141 0.01991 0.01233 0.02028 0.03281 0.05482]  (expected [0.01294 0.01141 0.01991 0.01233 0.02028 0.03281 0.05482]) |
| ✓ | published bias(tilt) a3a2 | [0.00747 0.03521 0.02873 0.0211  0.01934 0.01499 0.01598]  (expected [0.00747 0.03521 0.02873 0.0211  0.01934 0.01499 0.01598]) |
| ✓ | published bias(tilt) a3a1 | [0.00856 0.01181 0.01732 0.01255 0.01497 0.02467 0.03318]  (expected [0.00856 0.01181 0.01732 0.01255 0.01497 0.02467 0.03318]) |
| ✓ | published bias(tilt) b2b1 | [0.00745 0.03285 0.02665 0.01702 0.02755 0.02928 0.03388]  (expected [0.00745 0.03285 0.02665 0.01701 0.02755 0.02928 0.03388]) |
| ✓ | published bias(tilt) alpha | [-0.04998  0.07243  0.13747  0.11659  0.08736  0.10442  0.09285]  (expected [-0.04998  0.07243  0.13747  0.11659  0.08736  0.10442  0.09285]) |
| ✓ | corrected a2a1, dedup calibration | [46 values, max /diff/ 9.7e-17]  (expected ) |
| ✓ | corrected a3a2, dedup calibration | [46 values, max /diff/ 6.7e-11]  (expected ) |
| ✓ | corrected alpha, dedup calibration | [46 values, max /diff/ 3.4e-10]  (expected ) |
| ✓ | corrected a2a1, published calibration | [46 values, max /diff/ 9.7e-17]  (expected ) |
| ✓ | corrected a3a2, published calibration | [46 values, max /diff/ 1.1e-16]  (expected ) |
| ✓ | corrected alpha, published calibration | [46 values, max /diff/ 8.3e-17]  (expected ) |
| ✓ | forward model with a delta PSF reproduces the standard fit (chi2) 191130 | (expected 191130) |
| ✓ | ... and its slope | 0.607953  (expected 0.607953) |


## 03_common_shape.ipynb

| | Check | Result |
|---|---|---|
| ✓ | dedup f12 | 0.428981  (expected 0.428981) |
| ✓ | dedup sig_f12 | 0.0898738  (expected 0.0898738) |
| ✓ | dedup f23 | 0.748065  (expected 0.748065) |
| ✓ | dedup sig_f23 | 0.0520975  (expected 0.0520975) |
| ✓ | dedup common_a2a1 | 0.190636  (expected 0.190636) |
| ✓ | dedup common_a3a2 | 0.566416  (expected 0.566416) |
| ✓ | dedup median_alpha | 0.510648  (expected 0.510648) |
| ✓ | dedup n_prolate_f | 43  (expected 43) |
| ✓ | dedup n_exclude_isotropy_2sig | 44  (expected 44) |
| ✓ | dedup rho_f23_incl | 0.423867  (expected 0.423867) |
| ✓ | dedup C7_common_a2a1 | 0.226311  (expected 0.226311) |
| ✓ | dedup C7_common_a3a2 | 0.572101  (expected 0.572101) |
| ✓ | dedup C12_sigint_a2a1 | 0.0932569  (expected 0.0932569) |
| ✓ | dedup C12_sigint_a3a2 | 0.0702587  (expected 0.0702587) |
| ✓ | dedup C11_prolate_sigma | 7.58557  (expected 7.58557) |
| ✓ | dedup C11_frac_prolate | 0.934783  (expected 0.934783) |
| ✓ | dedup C10_n_consistent_isotropic | 4  (expected 4) |
| ✓ | dedup C13_n_collapsed | 3  (expected 3) |
| ✓ | dedup C7_rho_incl_b2b1 | -0.584829  (expected -0.584829) |
| ✓ | dedup C8_median_incl | 65.7336  (expected 65.7336) |
| ✓ | dedup C8_ks_p | 0.00567505  (expected 0.00567505) |
| ✓ | dedup a2/a1 upper limits | [0.11785 1.      0.61383]  (expected [0.11785 1.      0.61383]) |
| ✓ | published f12 | 0.436929  (expected 0.436929) |
| ✓ | published sig_f12 | 0.0863433  (expected 0.0863433) |
| ✓ | published f23 | 0.751789  (expected 0.751789) |
| ✓ | published sig_f23 | 0.0517969  (expected 0.0517969) |
| ✓ | published common_a2a1 | 0.19119  (expected 0.19119) |
| ✓ | published common_a3a2 | 0.565476  (expected 0.565476) |
| ✓ | published median_alpha | 0.500448  (expected 0.500448) |
| ✓ | published n_prolate_f | 43  (expected 43) |
| ✓ | published n_exclude_isotropy_2sig | 44  (expected 44) |
| ✓ | published rho_f23_incl | 0.426704  (expected 0.426704) |
| ✓ | published C7_common_a2a1 | 0.2244  (expected 0.2244) |
| ✓ | published C7_common_a3a2 | 0.569687  (expected 0.569687) |
| ✓ | published C12_sigint_a2a1 | 0.0939714  (expected 0.0939714) |
| ✓ | published C12_sigint_a3a2 | 0.0699008  (expected 0.0699008) |
| ✓ | published C11_prolate_sigma | 7.54935  (expected 7.54935) |
| ✓ | published C11_frac_prolate | 0.934783  (expected 0.934783) |
| ✓ | published C10_n_consistent_isotropic | 4  (expected 4) |
| ✓ | published C13_n_collapsed | 3  (expected 3) |
| ✓ | published C7_rho_incl_b2b1 | -0.589023  (expected -0.589023) |
| ✓ | published C8_median_incl | 65.7336  (expected 65.7336) |
| ✓ | published C8_ks_p | 0.00567505  (expected 0.00567505) |
| ✓ | published a2/a1 upper limits | [0.11612 1.      0.61383]  (expected [0.11612 1.      0.61383]) |


## 04_regions_and_depth.ipynb

| | Check | Result |
|---|---|---|
| ✓ | Region 2 drift (dU/dW, dV/dW) | [-0.63388  0.17713]  (expected [-0.63388  0.17713]) |
| ✓ | Region 2 coherent epochs | 4  (expected 4) |
| ✓ | Region 1 prof | [351 values, max /diff/ 0]  (expected ) |
| ✓ | Region 1 w_sep | [21 values, max /diff/ 0]  (expected ) |
| ✓ | Region 1 w_s2 | [21 values, max /diff/ 0]  (expected ) |
| ✓ | Region 1 aniso_min | 4.75808  (expected 4.75808) |
| ✓ | Region 1 aniso_max | 7.50187  (expected 7.50187) |
| ✓ | Region 1 w_over_max_median | 0.923679  (expected 0.923679) |
| ✓ | Region 1 w_over_mean_median | 1.28669  (expected 1.28669) |
| ✓ | Region 1 w_over_mean_min | 0.645622  (expected 0.645622) |
| ✓ | Region 2 prof | [351 values, max /diff/ 0]  (expected ) |
| ✓ | Region 2 w_sep | [21 values, max /diff/ 0]  (expected ) |
| ✓ | Region 2 w_s2 | [21 values, max /diff/ 0]  (expected ) |
| ✓ | Region 2 aniso_min | 1.17431  (expected 1.17431) |
| ✓ | Region 2 aniso_max | 1.19069  (expected 1.19069) |
| ✓ | Region 2 w_over_max_median | 0.569482  (expected 0.569482) |
| ✓ | Region 2 w_over_mean_median | 0.612633  (expected 0.612633) |
| ✓ | Region 2 w_over_mean_min | 0.386291  (expected 0.386291) |
| ✓ | C5 trackability R1, R2 box | [0.31512 0.85545]  (expected [0.31512 0.85545]) |
| ✓ | per-window ncct | [46 values, max /diff/ 0]  (expected ) |
| ✓ | per-window A | [46 values, max /diff/ 0]  (expected ) |
| ✓ | per-window l_min | [46 values, max /diff/ 0]  (expected ) |
| ✓ | per-window incl | [46 values, max /diff/ 0]  (expected ) |
| ✓ | rho(trackability, inclination) | -0.758495  (expected -0.758495) |
| ✓ | rho(trackability, anisotropy) | -0.0743139  (expected -0.0743139) |
| ✓ | rho(trackability, width) | 0.453962  (expected 0.453962) |
| ✓ | windows drifting toward -U | 45  (expected 45) |


## 05_science_figure.ipynb

| | Check | Result |
|---|---|---|
| ✓ | F: every bounded window drawn | 44  (expected 44) |
| ✓ | F: plotted a2/a1 (published calibration) | [43 values, max /diff/ 9.7e-17]  (expected ) |
| ✓ | F: plotted a3/a2 (published calibration) | [43 values, max /diff/ 1.1e-16]  (expected ) |
| ✓ | F: common shape (published) | [0.19119 0.56548]  (expected [0.19119 0.56548]) |
| ✓ | F: upper limits (published) | [0.11612 0.61383 1.     ]  (expected [0.11612 0.61383 1.     ]) |
| ✓ | A: Region 2 drift arrow | [-0.63388  0.17713]  (expected [-0.63388  0.17713]) |
| ✓ | C/E: Region 1 wedge profiles | [351 values, max /diff/ 0]  (expected ) |
| ✓ | C/E: Region 1 W points | [21 values, max /diff/ 0]  (expected ) |
| ✓ | C/E: Region 2 wedge profiles | [351 values, max /diff/ 0]  (expected ) |
| ✓ | C/E: Region 2 W points | [21 values, max /diff/ 0]  (expected ) |
| ✓ | figure size [in] | [7.25   6.3166]  (expected [7.25   6.3166]) |


## 06_sensitivity.ipynb

| | Check | Result |
|---|---|---|
| ✓ | paper row f12 = notebook 03 | 0.365327  (expected 0.365327) |
| ✓ | paper row f23 = notebook 03 | 0.752428  (expected 0.752428) |
| ✓ | paper row sig_f12_95 = notebook 03 | 0.116653  (expected 0.116653) |
| ✓ | paper row sig_f23_95 = notebook 03 | 0.106612  (expected 0.106612) |
| ✓ | paper row common shape = notebook 03 | [0.13849 0.57209]  (expected [0.13849 0.57209]) |
| ✓ | D24 row f12 = original analysis | 0.428981  (expected 0.428981) |
| ✓ | D24 row f23 = original analysis | 0.748065  (expected 0.748065) |
| ✓ | D24 row sig_f12 = original analysis | 0.0898738  (expected 0.0898738) |
| ✓ | D24 row sig_f23 = original analysis | 0.0520975  (expected 0.0520975) |
| ✓ | D24 row common shape = original analysis | [0.19064 0.56642]  (expected [0.19064 0.56642]) |
| ✓ | D24 row prolate sigma = original analysis | 7.58557  (expected 7.58557) |
| ✓ | D24 row median inclination = original analysis | 65.7336  (expected 65.7336) |
| ✓ | paper row C1-C4 = original regions | [4.75808 1.19069 0.92368 0.61263]  (expected [4.75808 1.19069 0.92368 0.61263]) |


## Known, accepted differences

* PSF injection calibration (`results/psf_calibration.csv`): 1321 of 1404 fitted rows are bit-identical to the original; the remaining 83 differ by at most 2e-5 in an axis ratio (0.0005 deg in inclination), from thread-dependent floating-point rounding in the optimiser.  The bias table and every corrected window value agree to < 1e-4 relative; all population numbers agree to the precision quoted.
* The original PSF bias table (`psf_calibration_published.csv`) contains duplicated synthetic fields (1368 rows, 78 distinct).  `Config.calib = 'dedup'` (default) uses each field once; `'published'` reproduces the first-submitted numbers.
