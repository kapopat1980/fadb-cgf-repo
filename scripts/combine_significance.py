#!/usr/bin/env python3
"""
Recomputes McNemar's test per seed from the *_full.json prediction files and
combines results across seeds via Fisher's method (with Stouffer's method as
a cross-check). This is the exact analysis used to produce Table 4 in the
manuscript, and was originally needed because the sweep's own
statistical_tests.json output was found corrupted mid-run - kept here as the
canonical, reproducible source of the significance results.

Usage:
    python scripts/combine_significance.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SEEDS, ALPHA
from src.stats_utils import mcnemar_test, fishers_method, stouffers_method

PAIRS = [
    ("fadb_cgf_gated", "efficientnet_baseline"),
    ("fadb_cgf_concat", "efficientnet_baseline"),
    ("fadb_cgf_gated", "resnet50_baseline"),
    ("fadb_cgf_concat", "resnet50_baseline"),
    ("fadb_cgf_gated", "fadb_cgf_concat"),
]


def load_full_result(model_name, seed):
    with open(RESULTS_DIR / f"{model_name}_seed{seed}_full.json") as f:
        return json.load(f)


def main():
    per_run_csv = RESULTS_DIR / "per_run_metrics.csv"
    df = pd.read_csv(per_run_csv) if per_run_csv.exists() else None

    combined_results = {}
    print(f"{'Comparison':<45}{'Seeds':<7}{'Fisher p':<12}{'Stouffer p':<12}{'Verdict'}")
    print("-" * 100)

    for model_a, model_b in PAIRS:
        key = f"{model_a}_vs_{model_b}"
        p_values = []
        for seed in SEEDS:
            try:
                ra = load_full_result(model_a, seed)
                rb = load_full_result(model_b, seed)
            except FileNotFoundError:
                continue
            _, _, p = mcnemar_test(ra["predictions"], rb["predictions"], ra["labels"])
            p_values.append(p)

        if not p_values:
            print(f"{key:<45} {'0':<7} {'n/a':<12} {'n/a':<12} no data")
            continue

        fisher_stat, fisher_p = fishers_method(p_values)
        stouffer_z, stouffer_p = stouffers_method(p_values)
        verdict = "SIGNIFICANT" if (fisher_p is not None and fisher_p < ALPHA) else "not significant"

        combined_results[key] = {
            "n_seeds": len(p_values), "individual_p_values": p_values,
            "fisher_statistic": fisher_stat, "fisher_combined_p": fisher_p,
            "stouffer_z": stouffer_z, "stouffer_combined_p": stouffer_p,
            "verdict_alpha_0.05": verdict,
        }
        fp_str = f"{fisher_p:.4f}" if fisher_p is not None else "n/a"
        sp_str = f"{stouffer_p:.4f}" if stouffer_p is not None else "n/a"
        print(f"{key:<45} {len(p_values):<7} {fp_str:<12} {sp_str:<12} {verdict}")

    out_path = RESULTS_DIR / "combined_significance.json"
    json.dump(combined_results, open(out_path, "w"), indent=2)
    print(f"\nSaved combined significance results to {out_path}")


if __name__ == "__main__":
    main()
