# The feasibility ceiling of literature-mined zeolite synthesis data — data and code release

This repository contains the dataset, the frozen splits, the trained model bundle and
the complete supplementary-analysis code for the diagnostic study

> *The feasibility ceiling of literature-mined zeolite synthesis data: a diagnostic study.*

No manuscript text is included; the repository holds only the artefacts listed in the
Data and Software Availability section of the paper.

## Contents

```
.
├── wp2_p22_supp_v0.py          # supplementary analyses S1–S4 (single entry point)
├── p22_figs_v0.py              # deterministic regeneration of Figures 3–5
├── l2_v0_1_litfail.py          # feature/protocol modules imported by the entry point
├── l3_purity_v1.py
├── feasibility_scorer_v1_2.py
├── comp_neg_knn.py
├── l1_baseline_v1.py
├── l1_feat_iter.py
├── osda_schema.py
├── expand_osda_desc.py
├── Data/
│   ├── zeosyn_db_v1_1.parquet              # literature-mined synthesis dataset
│   ├── zeosyn_splits_v1_1.parquet          # frozen split assignment
│   ├── zeosyn_db_dedup_manifest.csv        # deduplication manifest
│   ├── zeobench_db_v1_8.parquet            # reference benchmark database
│   ├── zeobench_splits_v1_8.parquet        # reference benchmark splits
│   ├── feasibility_scorer_v1_2_model.pkl   # frozen model bundle (LightGBM head)
│   ├── feasibility_scorer_v1_2_manifest.json
│   ├── m10_calib_v1_report.txt             # deployment calibration record
│   └── m10_calib_v1_manifest.json
└── Documentation/P2-2/
    ├── data/   # per-setting metrics file, run report and run manifest (S1–S4)
    └── assets/ # Figures 3–5 (300 dpi) and their md5 manifest
```

All released binaries carry md5 checksums. `Data/feasibility_scorer_v1_2_manifest.json`
records the model bundle hash; `Documentation/P2-2/data/wp2_p22_supp_v0_manifest.json`
records the code and input-data hashes of the run that produced the released metrics
file; `Documentation/P2-2/data/p22_figs_manifest.json` records the figure files.

## Requirements

Python 3.11+ with:

```
numpy, pandas, lightgbm, scikit-learn, scipy, rdkit, pyarrow
```

See `requirements.txt` for the versions used in the released run.

## Reproduction

See [REPRODUCE.md](REPRODUCE.md) for the command list. In short:

```
python wp2_p22_supp_v0.py    # ~1 min; aborts unless the anchor reading is reproduced
python p22_figs_v0.py        # regenerates Figures 3–5 from the metrics file
```

`wp2_p22_supp_v0.py` first re-derives the headline `xtal` test AUROC under the frozen
protocol and compares it against the registered anchor `0.6307095094096244`
(tolerance 1e-3, 36 boosting rounds). If the anchor is not reproduced the script
aborts without writing outputs. The directory layout of this repository mirrors the
one the scripts expect; both scripts resolve all paths relative to their own location.

## License

- Code: [MIT](LICENSE)
- Data (parquet/csv files under `Data/` and `Documentation/P2-2/data/`):
  [CC BY 4.0](LICENSE-DATA)

## Publishing this repository (maintainer notes)

The manuscript cites a repository URL and an archived DOI; both are placeholders until
submission. To publish:

1. **Create the remote repository.**
   - GitHub (or any Git host): create an empty repository, e.g.
     `<org>/zeolite-feasibility-ceiling`.
   - Do not initialise it with a README; this repository already contains one.
2. **Attach the remote and push.**

   ```
   git remote add origin git@github.com:<org>/zeolite-feasibility-ceiling.git
   git push -u origin main
   ```

   If the host rejects the 46 MB model bundle, verify the push limit (GitHub warns at
   50 MB and blocks at 100 MB) — the bundle is below both. Alternatively host the bundle
   via Git LFS and update the manifest accordingly; if you do, re-run
   `python wp2_p22_supp_v0.py` afterwards and confirm the anchor check still passes.
3. **Archive for the DOI.**
   - Create a release on the host (tag, e.g. `v1.0.0`, exactly the committed tree).
   - Deposit the release on Zenodo (GitHub–Zenodo integration or manual upload of the
     release archive). Reserve the DOI and record it in the release notes.
4. **Fill in the manuscript.** Replace the placeholder in the Data and Software
   Availability section with the repository URL and the archived DOI, and verify the
   anchor command from `REPRODUCE.md` on a clean clone before submission.
