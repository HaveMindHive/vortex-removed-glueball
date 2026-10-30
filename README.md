# The scalar glueball on center-vortex-removed SU(2) lattice ensembles: a data note

Author: Turaq Rayn-McLean. Draft, not submitted. License: CC BY 4.0 (see LICENSE).

`paper.md` / `paper.pdf`: a short hep-lat note. It measures the 0++ glueball correlator and the static potential on SU(2)
Wilson-action ensembles at beta 2.3, 2.4, 2.5, on the original links (ORIG), the center-projected Z2 links in direct
maximal center gauge (CP), and the vortex-removed links (VR, de Forcrand-D'Elia). Clean result at beta 2.4 (VR mass
volume independent, 2.40(8) sqrt(sigma), string tension zero within errors); at beta 2.5 the pre-registered string-tension
premise check fails with a short-time estimator and a gauge-copy study does not remove it; the pre-registered combined
outcome is inconclusive. Limitations are listed first in the paper.

## Contents
- `data/series1_summary.csv`: beta 2.3 (8^4, 12^4, 16^4) and 2.4 (12^4, 16^4), T = L. Per (beta, L, ensemble, basis):
  measurements, streams, tau_int, plaquette, P-vortex density, a^2 sigma (T 3->4, free Coulomb term) and error, effective
  masses m(1..3), quoted mass and plateau flag, a sqrt(sigma_ORIG) unit, mass in units of sqrt(sigma_ORIG).
- `data/series2_summary.csv`: beta 2.3 (12^3x32, 16^3x32) and 2.5 (16^4, 20^4), same columns plus m(1..6) and a
  report-only late plateau.
- `data/series2_long_range_estimators.csv`: sigma_X / sigma_ORIG with E1, E1' (R >= 4), E2 (T = 5, 6), Creutz ratios, on
  the series 2 configurations.
- `data/series3_gauge_copies.csv`, `data/series3_statistics.csv`: gauge-copy study (single copy vs best of 16, paired):
  MCG functional, every string-tension estimator ratio, E1 ratio by copy rank, glueball masses.
- `data/ratios_and_volume_tests.csv`: headline ratios, volume tests and their pre-registered calls.
- `code/vr_glueball.py`: simulation (heat bath + over-relaxation), direct maximal center gauge (numba), CP/VR split,
  glueball operators, Wilson loops, and the series 2 analysis (jackknife, GEVP, plateau rule, volume tests).
- `code/gauge_copies.py`: series 3 (imports `vr_glueball.py`): random gauge copies, paired measurement, estimators.

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
(one 16^4 measurement with 16 gauge copies takes about 4 min); series 1 ran 11 streams for 6 h each. Each analysis takes a
few minutes. Raw configurations and per-measurement files (several GB) are not included; the CSVs hold every quantity
used in the paper.

Checks: run on the original series 2 measurement files, the cleaned `vr_glueball.py analyze` reproduces
`data/series2_summary.csv` exactly (all 24 rows). The printed "combined verdict" in that script is the series 2 rule as
written, which does not apply the string-tension premise check; the settled outcome is in sec. 5 of the paper.

Series 1 (beta 2.3 / 2.4, T = L) was produced by an earlier version of the same pipeline (numpy instead of numba for the
gauge fixing, same algorithm; the numba version is asserted equal to the numpy version in `selftest`); that script is not
included because it depends on earlier exploratory files. Its numbers are in `data/series1_summary.csv`, and the included
code can generate the same ensembles (`run 16 16 2.4 <seed> <deadline>`).

## How this was made
Claude (Anthropic), used as a research assistant through Claude Code, wrote and ran the simulation and analysis code,
searched and read the literature, and drafted the text under the author's direction. The author chose the questions,
reviewed the work and is responsible for the content.

## Citation
See CITATION.cff.
