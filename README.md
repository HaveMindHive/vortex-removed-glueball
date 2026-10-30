# The 0++ and 2++ glueball channels on center-vortex-removed SU(2) lattice ensembles: a data note

Author: Turaq Rayn-McLean. Version 1.2: draft, revised after referee review and a fourth data series, not submitted. License: text and data CC BY 4.0 (see
LICENSE); code in `code/` MIT (see code/LICENSE).

`paper.md` / `paper.pdf`: a short hep-lat note. It measures the 0++ and 2++ glueball correlators and the static
potential on SU(2) Wilson-action ensembles at beta 2.3, 2.35, 2.4, 2.5, on the original links (ORIG), the center-projected
Z2 links in direct maximal center gauge (CP), and the vortex-removed links (VR, de Forcrand-D'Elia). Removing the vortices
leaves a nonzero 0++ decay rate at beta 2.35, 2.4 and 2.5, but its ratio to the original glueball mass is not robust: it
rises by 2.3 sigma with better gauge copies at beta 2.4 (best of 8) and 2.5 (best of 16), and it differs by 3.4 sigma between
beta 2.35 (0.61(4)) and 2.4 (0.78(3)), the pre-registered series 4 outcome "fraction not robust". What survives: VR string
tension zero at the largest distances resolved (not at intermediate distances), a volume-independent single-copy VR mass at
beta 2.4, and a VR 2++ above the free two-gluon box level that is not a scaled copy of the original 2++. Limitations are
listed first in the paper.

## Contents
- `data/series1_summary.csv`: beta 2.3 (8^4, 12^4, 16^4) and 2.4 (12^4, 16^4), T = L. Per (beta, L, ensemble, basis):
  measurements, streams, tau_int, plaquette, P-vortex density, a^2 sigma (T 3->4, free Coulomb term) and error, effective
  masses m(1..3), quoted mass and plateau flag, a sqrt(sigma_ORIG) unit, mass in units of sqrt(sigma_ORIG).
- `data/series2_summary.csv`: beta 2.3 (12^3x32, 16^3x32) and 2.5 (16^4, 20^4), same columns plus m(1..6) and a
  report-only late plateau.
- `data/series2_long_range_estimators.csv`: sigma_X / sigma_ORIG with E1, E1' (R >= 4), E2 (T = 5, 6) and the full VR
  Creutz-ratio series chi(R,R), R = 2..L/2, on the series 2 configurations (`gauge_copies.py reanalyze_series2`).
- `data/series1_beta2.4_long_time_estimators.csv`: the same estimators on the stored beta 2.4 series 1 Wilson loops, VR and
  CP, with the ORIG values (`series1_vr_glueball.py long24`). Analysis done after referee review, not pre-registered.
- `data/bin_size_scan.csv`: jackknife errors of the VR m(1), m(2) (basis A) for 4 to 40 blocks per stream, series 1 and 2.
- `data/series1_mcg_functional.csv`: mean MCG functional before and after gauge fixing, series 1.
- `data/series3_gauge_copies.csv`, `data/series3_statistics.csv`: gauge-copy study (single copy vs best of 16, paired):
  MCG functional, every string-tension estimator ratio, E1 ratio by copy rank, glueball masses.
- `data/series4_summary.csv`: series 4 (beta 2.4 12^4, beta 2.35 10^4, and the series 1 beta 2.4 configurations as data
  set D-S1). Per (data set, beta, L, ensemble, channel 0++ / 2++, GEVP level 0 / 1): measurements, effective masses m(1..3)
  and errors, quoted mass and plateau flag. Ensemble VRn = vortex-removed with the best of 8 gauge copies (data set D-copy).
  Produced by `code/vr2_spectrum.py analyze`.
- `data/series4_analyze.log`: the full series 4 analysis output (string-tension ratios, mass ratios, paired gauge-copy
  differences, and the pre-registered decisions), written by the same run.
- `data/ratios_and_volume_tests.csv`: headline ratios, volume tests and their pre-registered calls, plus report-only rows
  (literature-unit masses, the m(2) volume ratio at beta 2.4, the beta 2.3 gauge-copy mass shift).
- `code/vr_glueball.py`: simulation (heat bath + over-relaxation), direct maximal center gauge (numba), CP/VR split,
  glueball operators, Wilson loops, and the series 2 analysis (jackknife, GEVP, plateau rule, volume tests).
- `code/gauge_copies.py`: series 3 (imports `vr_glueball.py`): random gauge copies, paired measurement, estimators.
- `code/vr2_spectrum.py`: series 4 (imports `gauge_copies.py`, which imports `vr_glueball.py`): plane-resolved loop
  operators (0++ = A1, 2++ = E), GEVP levels 0 and 1, optional paired best-of-N gauge copies, beta 2.35 smearing scale.
- `code/series1_vr_glueball.py`: series 1 (beta 2.3 / 2.4, T = L): simulation (numpy gauge fixing), the series 1 analysis,
  and the post-referee beta 2.4 long-time reanalysis (`long24`, imports the estimators of `gauge_copies.py`).

## Reproduce
Requirements: Python 3.11+, numpy, numba, matplotlib.

```
cd code
python3 vr_glueball.py selftest                      # a few seconds
python3 gauge_copies.py selftest                     # under a minute
# one stream (deadline is a Unix timestamp); series 2 used e.g. 16 16 2.5 351 and 20 20 2.5 361/362
python3 vr_glueball.py run 16 16 2.5 351 $(( $(date +%s) + 6*3600 ))
python3 vr_glueball.py analyze                       # writes runs/summary.csv, runs/analyze.log, PNGs
python3 gauge_copies.py run 16 16 2.5 451 $(( $(date +%s) + 6*3600 ))
python3 gauge_copies.py analyze                      # writes runs_gauge_copies/summary.csv, analyze.log, PNG
```
Seeds used: series 2: 301, 302 (beta 2.3, 16^3x32), 311 (12^3x32), 351 (2.5, 16^4), 361, 362 (20^4). Series 3: 451-458
(16^4), 461, 462 (20^4), 411, 412 (2.3, 12^3x32).

Runtime (one core per stream, Apple silicon): series 2 ran 6 streams for 6 h each; series 3 ran 12 streams for 6 h each
(one 16^4 measurement with 16 gauge copies takes about 4 min); series 1 ran 11 streams to one deadline (8 for 6 h, 3 added later for about 3.6 h). Each analysis takes a
few minutes. Raw configurations and per-measurement files (several GB) are not included; the CSVs hold every quantity
used in the paper.

Checks: run on the original series 2 measurement files, the cleaned `vr_glueball.py analyze` reproduces
`data/series2_summary.csv` exactly (all 24 rows). The printed "combined verdict" in that script is the series 2 rule as
written, which does not apply the string-tension premise check; the settled outcome is in sec. 5 of the paper.

Series 1 (beta 2.3 / 2.4, T = L) was produced by `code/series1_vr_glueball.py`, an earlier version of the same pipeline
(numpy instead of numba for the gauge fixing, same algorithm; the numba version is asserted equal to the numpy version in
`vr_glueball.py selftest`). It reads production streams from `code/runs_series1/` and the earlier exploratory streams
(beta 2.3 8^4 and 12^4 seeds 1, 2; beta 2.4 12^4 seeds 1, 2; written by an earlier script with three smearing levels) from
`code/runs_series1/exploratory/`. Run on the original measurement files, `series1_vr_glueball.py analyze` reproduces
`data/series1_summary.csv` exactly (all 28 rows), and `long24` writes `data/series1_beta2.4_long_time_estimators.csv`.
Series 1 seeds: 101-105 (beta 2.3, 16^4), 201-204 (2.4, 16^4), 111 (2.3, 12^4), 211 (2.4, 12^4).
```
python3 series1_vr_glueball.py selftest
python3 series1_vr_glueball.py run 16 2.4 201 $(( $(date +%s) + 6*3600 ))
python3 series1_vr_glueball.py analyze
python3 series1_vr_glueball.py long24
```
Series 4: seeds 701 (beta 2.4, 12^4, best of 8 copies, `ncopy` = 7), 702 (2.4, 12^4, single copy), 711, 712 (2.35,
10^4, single copy); four streams to one deadline, 2.5 h. Run on the original measurement files (placed in `code/runs_vr2/`,
with the series 1 files in `code/runs_series1/`), `vr2_spectrum.py analyze` reproduces `data/series4_summary.csv` exactly
(all 26 rows).
```
python3 vr2_spectrum.py selftest                     # a few seconds
python3 vr2_spectrum.py run 12 2.4 701 $(( $(date +%s) + 9000 )) 7
python3 vr2_spectrum.py run 10 2.35 711 $(( $(date +%s) + 9000 )) 0
python3 vr2_spectrum.py analyze                      # writes runs_vr2/summary.csv, runs_vr2/analyze.log
```
The pre-registration texts were committed to a private git repository before each series' production data existed; they
are internally time-stamped, not externally registered.

## How this was made
Claude (Anthropic), used as a research assistant through Claude Code, wrote and ran the simulation and analysis code,
searched and read the literature, and drafted the text under the author's direction. The author chose the questions,
reviewed the work and is responsible for the content.

## Citation
See CITATION.cff.
