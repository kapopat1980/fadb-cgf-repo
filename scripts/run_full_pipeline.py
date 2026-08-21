#!/usr/bin/env python3
"""
Runs the full experiment pipeline: leakage audit, then the multi-seed
training sweep across all 5 model variants.

Usage:
    python scripts/run_full_pipeline.py

This mirrors notebooks/fadb_cgf_full_pipeline.ipynb but as a plain script
suitable for local or HPC execution. For Kaggle, use the notebook directly
(GPU quota / session-length considerations documented there apply).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.config import (
    TRAIN_DIR, TEST_DIR, CKPT_DIR, RESULTS_DIR, SEEDS, get_model_registry,
)
from src.data import BrainTumorDataset, train_transform, eval_transform, make_loaders
from src.leakage_detection import run_leakage_pipeline
from src.train import train_one_run
from src.evaluate import evaluate_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(exist_ok=True)

    # ---- Step 1: leakage audit ----
    leak_cache = RESULTS_DIR / "leakage_ssim_verified.json"
    if leak_cache.exists():
        print(f"Loading cached leakage results from {leak_cache}")
        cache = json.load(open(leak_cache))
        verified_leaks = cache["verified_leaks"]
    else:
        print("Running two-stage leakage detection pipeline...")
        verified_leaks, rejected, n_test = run_leakage_pipeline(TRAIN_DIR, TEST_DIR)
        json.dump({"verified_leaks": verified_leaks, "rejected_false_positives": rejected},
                   open(leak_cache, "w"), indent=2)
        print(f"Confirmed {len(verified_leaks)} leaks out of {n_test} test images "
              f"({100*len(verified_leaks)/n_test:.1f}%)")

    leak_paths = {row[0] for row in verified_leaks}

    # ---- Step 2: build datasets ----
    train_dataset = BrainTumorDataset(TRAIN_DIR, train_transform)
    test_dataset = BrainTumorDataset(TEST_DIR, eval_transform, exclude_files=leak_paths)
    print(f"Train size: {len(train_dataset)}, Test size (post leakage filter): {len(test_dataset)}")

    # ---- Step 3: multi-seed sweep ----
    MODEL_REGISTRY = get_model_registry()
    results_csv_path = RESULTS_DIR / "per_run_metrics.csv"
    rows = pd.read_csv(results_csv_path).to_dict("records") if results_csv_path.exists() else []
    completed = {(r["model_name"], r["seed"]) for r in rows}

    for model_name, model_fn in MODEL_REGISTRY.items():
        for seed in SEEDS:
            if (model_name, seed) in completed:
                print(f"Skipping {model_name} seed {seed} (already completed)")
                continue
            print(f"\n=== Training {model_name} | seed {seed} ===")
            model, _ = train_one_run(model_fn, model_name, seed, train_dataset, test_dataset, device)
            _, _, test_loader = make_loaders(train_dataset, test_dataset, seed)
            result = evaluate_model(model, test_loader, device)

            rows.append({"model_name": model_name, "seed": seed, "accuracy": result["accuracy"],
                         "macro_f1": result["macro_f1"], "gm_confusion": result["glioma_meningioma_confusion"]})
            pd.DataFrame(rows).to_csv(results_csv_path, index=False)

            import numpy as np
            np.save(RESULTS_DIR / "confusion_matrices" / f"{model_name}_seed{seed}.npy",
                    np.array(result["confusion_matrix"]))
            json.dump(result, open(RESULTS_DIR / f"{model_name}_seed{seed}_full.json", "w"))

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nPipeline complete. See", results_csv_path)


if __name__ == "__main__":
    main()
