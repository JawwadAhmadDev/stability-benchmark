# Stability Over Detection Performance — code and results

Code and aggregated results for:

> M. Majid, T. Ali, J. Ahmad, and G. S. Choi, "Stability Over Detection Performance: Benchmarking Thirty Feature-Selection Methods Across Four Healthcare-Security Datasets," submitted to *IEEE Access*, 2026.

<!-- After the Zenodo release, paste the DOI badge here, e.g. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

The study compares thirty feature-selection methods from four families (filters, wrappers, embedded methods, and hybrids) on four healthcare-security datasets, scoring each method on detection performance (AUC and MCC), selection stability (Nogueira and Jaccard indices over 30 bootstrap resamples), and compactness (number of selected features).

## Repository structure

```
notebooks/   NB0–NB4, the analysis pipeline (run in order)
scripts/     make_decision_table_and_figures.py
results/     aggregated results released with the paper
figures/     the paper's figures, as produced by the script
```

## Notebooks (run in this order)

| Notebook | Name in the paper | What it does | Main outputs |
|---|---|---|---|
| `NB0_HIPAA_feature_construction.ipynb` | HIPAA feature-construction notebook | Builds the 32 numeric HIPAA features from the HHS OCR breach-portal export (HIPAA only) | `processed/feature_engineered.csv` |
| `NB1_data_preparation.ipynb` | NB1 | Subsampling, leakage audit, stratified 80/20 split, train-fitted imputation and min-max scaling, for all four datasets | `processed/train.parquet`, `processed/test.parquet`, `processed/meta.json` |
| `NB2_feature_selection.ipynb` | NB2 | Runs the 30 feature-selection methods on each training split | `results/nb2_selection_matrix.csv` and manifests |
| `NB3_evaluation_stability.ipynb` | NB3 | Classifier selection by training-only CV, held-out test evaluation (AUC, MCC and supplementary metrics), 30-bootstrap selection stability, compactness | `results/nb4_*.csv`, `results/nb4_master_audit.json` |
| `NB4_cross_dataset_analysis.ipynb` | NB4 | Average ranks, Friedman and Iman-Davenport tests, Nemenyi critical differences, Kendall's W, family-level analysis, trade-offs, figures | `_aggregate_master_30x4_v12/tables/`, `.../figures/` |

**Output file prefixes.** NB3 and NB4 were developed under the names "Notebook 4" and "Notebook 5". Their output files keep the prefixes `nb4_` and `nb5_` so that they match the released result files: `nb4_*` files are written by NB3, and `nb5_*` files by NB4.

## Data

The datasets are public and are **not redistributed** here. Download them from:

- **HIPAA breach records:** U.S. Department of Health and Human Services, Office for Civil Rights, breach portal, https://ocrportal.hhs.gov/ocr/breach/ (reports from January 2021 to May 2026)
- **WUSTL-EHMS-2020:** https://www.cse.wustl.edu/~jain/ehms/
- **CICIoMT2024:** https://www.unb.ca/cic/datasets/iomt-dataset-2024.html (Wi-Fi/MQTT test file)
- **WUSTL-HDRL-2024:** https://www.cse.wustl.edu/~jain/hdrl/

Expected folder layout (set `BENCH_ROOT` in the configuration cell of NB1–NB4):

```
BENCH_ROOT/
├── HIPAA/data/        <- copy NB0's processed/feature_engineered.csv here
├── WUSTL/data/        <- WUSTL-EHMS-2020 CSV
├── CICIOMT/data/      <- CICIoMT2024 Wi-Fi/MQTT test CSV
└── WUSTL_HDRL/data/   <- WUSTL-HDRL-2024 CSV
```

NB1 creates `processed/`, `features/`, `eval/` and `results/` inside each dataset folder. NB0 was written for Google Colab (it mounts Google Drive); to run it locally, set its data paths in the configuration cell.

## Environment

Python 3.13 with the packages in `requirements.txt`. All randomness is fixed by a single seed (`RANDOM_STATE = 42`). NB2 and NB3 checkpoint their progress and resume from completed work.

## Rebuilding the decision table and figures

```
python scripts/make_decision_table_and_figures.py
```

The script reads `results/nb5_master_long_form_30x4.csv` and writes `results/decision_table_30methods.csv` and the folder `figures/`, which holds all eight figures of the paper (PNG at 600 dpi and vector PDF):

| File | Figure in the paper |
|---|---|
| `Fig1_methodology` | Fig. 1, end-to-end methodology |
| `Fig2_CD_diagrams` | Fig. 2, critical-difference diagrams (AUC, MCC, Nogueira) |
| `Fig3_stability_by_method` | Fig. 3, mean Nogueira index per method |
| `Fig4_family_ranks` | Fig. 4, average ranks of the four families |
| `Fig5_performance_vs_stability` | Fig. 5, mean AUC against mean Nogueira index |
| `Fig6_outcome_correlations` | Fig. 6, Spearman correlations between the five outcomes |
| `Fig7_stability_rank_per_dataset` | Fig. 7, stability rank of each method on each dataset |
| `Fig8_nogueira_vs_jaccard` | Fig. 8, mean Nogueira index against mean Jaccard index |
 It also prints the checks reported in the paper: the Nemenyi critical difference (23.33), the compactness weight below which the selected method stays selected (0.285), and the Spearman correlations between mean AUC and mean Nogueira index (−0.425, p = 0.019) and between mean Jaccard and mean Nogueira index (0.626, p = 0.0002). Ties in stability rank are broken by compactness rank, as in the paper's decision rule.

## Released results (`results/`)

- `decision_table_30methods.csv`: per-method mean scores, average ranks, and stability-first decision scores (30 methods), as written by the script above.
- `nb5_master_long_form_30x4.csv`: AUC, MCC, Nogueira, Jaccard, and compactness for all 30 methods on all 4 datasets.
- `stability/nb4_stability_<DATASET>.csv`: per-dataset stability results (30 methods × 30 bootstrap resamples).

## Citation and license

Please cite the article above and this archive (see `CITATION.cff`). The code is released under the MIT License (see `LICENSE`).
