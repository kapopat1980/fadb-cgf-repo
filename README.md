# FADB-CGF: Frequency-Aware Dual-Branch Network with Confidence-Gated Fusion

Code, trained-model artifacts, and reproducibility materials for:

> Meva, D., Popat, K. "Train–Test Leakage in a Widely Used Brain Tumor MRI Benchmark: A Quantified Audit and a Rigorous Multi-Seed Evaluation of Frequency-Aware Dual-Branch Fusion." *[Journal name, under review]*.

This repository accompanies the manuscript submitted for peer review and is provided to support reproducibility, in line with the paper's Data and Code Availability statements.

## Summary

This study makes two contributions on the four-class (glioma, meningioma, pituitary, no-tumor) Kaggle brain tumor MRI benchmark:

1. **A quantified train–test leakage audit.** A two-stage perceptual-hash-plus-SSIM pipeline found that **24.2%** of the benchmark's test set (**69.2%** of the "no tumor" class specifically) consists of confirmed near-duplicates of training images — a risk created by the dataset's construction (a merge of three source collections, rebalanced via augmentation) that no prior study using this dataset variant appears to have audited.
2. **A rigorous, multi-seed evaluation of FADB-CGF**, a proposed dual-branch architecture fusing a spatial (EfficientNetB0) branch with a frequency-domain (2D-FFT) branch, via two fusion mechanisms (concatenation and confidence-gated cross-attention). Neither fusion variant improves on a well-tuned single-backbone baseline on the leakage-corrected test set — reported as a transparent negative result.

See the manuscript for full methodology, results, and discussion.

## Repository Structure

```
.
├── src/                        # Core package: models, data, training, evaluation
│   ├── config.py                # Paths, hyperparameters, class definitions
│   ├── data.py                  # Dataset class, transforms, FFT preprocessing
│   ├── leakage_detection.py     # Two-stage pHash + SSIM leakage audit pipeline
│   ├── models.py                # SpatialBranch, FrequencyBranch, fusion heads, FADB_CGF
│   ├── train.py                 # Training loop with checkpoint resumption
│   ├── evaluate.py               # Accuracy, macro-F1, confusion matrix, per-class metrics
│   ├── robustness.py            # Noise/blur/brightness perturbation evaluation
│   └── stats_utils.py           # McNemar's test, Fisher's/Stouffer's p-value combination
├── scripts/                     # Executable entry points
│   ├── run_full_pipeline.py     # Leakage audit + full multi-seed training sweep
│   ├── run_robustness_eval.py   # Robustness evaluation (Section 5.5 of the manuscript)
│   └── combine_significance.py  # Recomputes and combines McNemar's tests across seeds
├── notebooks/
│   └── fadb_cgf_full_pipeline.ipynb   # Kaggle-ready notebook version of the full pipeline
├── figures/                     # Manuscript figures (architecture, pipeline, results)
├── results/                     # Small, version-controllable output artifacts
│   ├── leakage_check.json           # Stage 1 (pHash) candidate detection results
│   ├── leakage_ssim_verified.json   # Stage 2 (SSIM) verification results
│   ├── per_run_metrics.csv          # Accuracy/F1/confusion count per (model, seed)
│   ├── summary_table.csv            # Aggregated mean ± std per model
│   ├── combined_significance.json   # McNemar's + Fisher's/Stouffer's combined results
│   └── confusion_matrices/          # Per-run confusion matrices (.npy)
├── tests/                       # Basic sanity tests for core components
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

**Not included in this repository** (too large for version control): trained model checkpoints (`.pt` files, ~5–24 MB each × 20 runs) and the raw dataset. See "Reproducing the Experiments" below for how to obtain both. Checkpoints and full per-seed prediction files are available from the corresponding author on reasonable request, per the manuscript's Data Availability statement.

## Dataset

This study uses a balanced/augmented 7,200-image variant of the Kaggle "Brain Tumor MRI Dataset" (glioma, meningioma, pituitary, no-tumor; 1,400 training + 400 test images per class), itself a merge of the Figshare CE-MRI, SARTAJ, and Br35H collections. This is distinct from the more commonly cited 7,023-image release used in several comparison studies (see manuscript Section 5.1) — **verify you have the correct variant** before attempting to reproduce reported numbers exactly; class counts should match Table 1 of the manuscript.

Download the dataset from Kaggle and place it so that the directory structure matches:
```
data/brain-tumor-mri-dataset/
├── Training/{glioma,meningioma,notumor,pituitary}/
└── Testing/{glioma,meningioma,notumor,pituitary}/
```
Update `DATA_ROOT` in `src/config.py` if you use a different location.

## Installation

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and a CUDA-capable GPU for training in reasonable time (the full sweep is 5 model variants × 4 seeds = 20 runs; see manuscript Section 3.5 for training configuration).

## Reproducing the Experiments

### Option A: Kaggle (recommended for GPU access)
Upload `notebooks/fadb_cgf_full_pipeline.ipynb` to Kaggle, attach the dataset described above, enable a GPU accelerator, and run cells in order. The notebook includes preflight checks and is resumable across sessions (see in-notebook comments regarding Kaggle's 12-hour session limit and `/kaggle/working` persistence).

### Option B: Local / HPC
```bash
# 1. Leakage audit + full training sweep (resumable - safe to re-run)
python scripts/run_full_pipeline.py

# 2. Robustness evaluation (requires checkpoints from step 1)
python scripts/run_robustness_eval.py

# 3. Statistical significance testing
python scripts/combine_significance.py
```

All scripts are resumable: re-running `run_full_pipeline.py` skips any `(model, seed)` pair with a completed checkpoint, so an interrupted run can simply be restarted.

## Reproducing Figures

The four manuscript figures (`figures/`) were generated from the result artifacts in `results/`. Figure-generation code is not currently packaged as a standalone script in this repository; the underlying data (confusion matrices, per-run metrics) needed to regenerate them is included in `results/`.

## Key Results (see manuscript for full detail and statistical testing)

| Model | Test Accuracy | Macro-F1 |
|---|---|---|
| EfficientNetB0 (baseline) | 94.39% ± 0.35% | 93.78% |
| FADB-CGF (concat) | 94.08% ± 0.26% | 93.52% |
| ResNet50 (baseline) | 94.04% ± 0.12% | 93.35% |
| FADB-CGF (gated) | 94.02% ± 0.34% | 93.25% |
| Frequency-only (ablation) | 75.47% ± 1.10% | 74.15% |

All test-set metrics are computed on the **leakage-corrected** test set (1,213 of the original 1,600 images; see `results/leakage_ssim_verified.json` for the full audit trail).

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`) and, if applicable, the original dataset sources (Figshare CE-MRI, SARTAJ, Br35H — see manuscript references).

## License

Code in this repository is released under the MIT License (see `LICENSE`). The dataset is subject to its own original licensing terms on Kaggle and is not redistributed here.

## Contact

Divyakant Meva, Kalpesh Popat — Faculty of Computer Applications, Marwadi University, Rajkot, Gujarat, India.
[Corresponding author email to be added]
