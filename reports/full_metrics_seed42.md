# Four-quadrant metrics summary

Independent paired runs: 1

| Metric | SFT (mean ± sd) | MAPPO (mean ± sd) | Δ MAPPO−SFT |
|---|---|---|---|
| Benign accept (ID) | 0.233 ± 0.000 | 0.150 ± 0.000 | -0.083 |
| Benign accept (OOD) | 0.033 ± 0.000 | 0.033 ± 0.000 | +0.000 |
| Injection detect (ID) | 0.867 ± 0.000 | 0.883 ± 0.000 | +0.017 |
| Injection detect (OOD) | 0.683 ± 0.000 | 0.750 ± 0.000 | +0.067 |
| Attack success rate | 0.225 ± 0.000 | 0.183 ± 0.000 | -0.042 |
| Secret extraction rate | 0.013 ± 0.000 | 0.013 ± 0.000 | +0.000 |
| False-positive rate (benign) | 0.858 ± 0.000 | 0.900 ± 0.000 | +0.042 |
| False-negative rate (attack) | 0.225 ± 0.000 | 0.183 ± 0.000 | -0.042 |
| OOD benign rejection | 0.950 ± 0.000 | 0.950 ± 0.000 | +0.000 |
| OOD injection detection | 0.683 ± 0.000 | 0.750 ± 0.000 | +0.067 |
| Accuracy (ID) | 0.550 ± 0.000 | 0.517 ± 0.000 | -0.033 |
| Accuracy (OOD) | 0.358 ± 0.000 | 0.392 ± 0.000 | +0.033 |
| Accuracy (overall) | 0.454 ± 0.000 | 0.454 ± 0.000 | +0.000 |
| Precision (DEFEND) | 0.474 ± 0.000 | 0.476 ± 0.000 | +0.001 |
| Recall (DEFEND) | 0.775 ± 0.000 | 0.817 ± 0.000 | +0.042 |
| F1 (DEFEND) | 0.589 ± 0.000 | 0.601 ± 0.000 | +0.013 |
| Measured decision diversity | 0.125 ± 0.000 | 0.167 ± 0.000 | +0.042 |

Note: rates are aggregated over draws, not cases; sd is across independent paired training seeds when more than one run is supplied, otherwise 0 by construction.
