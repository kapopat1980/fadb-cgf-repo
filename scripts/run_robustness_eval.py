#!/usr/bin/env python3
"""
Runs the robustness evaluation (Gaussian noise, blur, brightness perturbation)
across all trained model checkpoints and seeds. Requires run_full_pipeline.py
to have been run first (or equivalent checkpoints present in outputs/checkpoints/).

Usage:
    python scripts/run_robustness_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.config import TRAIN_DIR, TEST_DIR, CKPT_DIR, RESULTS_DIR, SEEDS, get_model_registry
from src.data import BrainTumorDataset, train_transform, eval_transform, make_loaders
from src.robustness import evaluate_robustness
import json


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    leak_cache = RESULTS_DIR / "leakage_ssim_verified.json"
    leak_paths = set()
    if leak_cache.exists():
        cache = json.load(open(leak_cache))
        leak_paths = {row[0] for row in cache["verified_leaks"]}

    train_dataset = BrainTumorDataset(TRAIN_DIR, train_transform)
    test_dataset = BrainTumorDataset(TEST_DIR, eval_transform, exclude_files=leak_paths)

    MODEL_REGISTRY = get_model_registry()
    robustness_rows = []

    for model_name, model_fn in MODEL_REGISTRY.items():
        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"Skipping {model_name} seed {seed} - no checkpoint found")
                continue
            model = model_fn().to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            _, _, test_loader = make_loaders(train_dataset, test_dataset, seed)
            rob = evaluate_robustness(model, test_loader, device)
            rob["model_name"] = model_name
            rob["seed"] = seed
            robustness_rows.append(rob)
            print(model_name, seed, rob)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    robustness_df = pd.DataFrame(robustness_rows)
    out_path = RESULTS_DIR / "robustness_results.csv"
    robustness_df.to_csv(out_path, index=False)
    print(f"\nSaved robustness results to {out_path}")
    if len(robustness_df):
        print(robustness_df.groupby("model_name")[["clean", "noise", "blur", "brightness"]].mean())


if __name__ == "__main__":
    main()
