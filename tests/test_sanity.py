"""
Basic sanity tests - not exhaustive, but verify the core package imports and
runs without a GPU or dataset present (useful as a quick CI/local check).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def test_models_forward_pass():
    """Uses pretrained=False to avoid requiring network access for weight
    downloads during testing - this only verifies architecture correctness
    (shapes, wiring), not pretrained-weight loading."""
    from src.models import BaselineModel, FrequencyOnlyModel, FADB_CGF

    batch_size = 2
    spatial_x = torch.randn(batch_size, 3, 224, 224)
    freq_x = torch.randn(batch_size, 1, 224, 224)

    for fusion in ["concat", "gated"]:
        model = FADB_CGF(backbone="efficientnet_b0", fusion=fusion, pretrained=False)
        out = model(spatial_x, freq_x)
        assert out.shape == (batch_size, 4), f"FADB_CGF ({fusion}) output shape mismatch: {out.shape}"

    baseline = BaselineModel("efficientnet_b0", pretrained=False)
    out = baseline(spatial_x)
    assert out.shape == (batch_size, 4)

    freq_only = FrequencyOnlyModel()
    out = freq_only(spatial_x, freq_x)
    assert out.shape == (batch_size, 4)

    print("test_models_forward_pass: PASSED")


def test_fft_magnitude_shape():
    from src.data import compute_fft_magnitude
    from PIL import Image

    img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype(np.uint8))
    freq = compute_fft_magnitude(img, size=224)
    assert freq.shape == (1, 224, 224), f"Unexpected FFT tensor shape: {freq.shape}"
    assert freq.min() >= 0.0 and freq.max() <= 1.0, "FFT magnitude not normalized to [0,1]"

    print("test_fft_magnitude_shape: PASSED")


def test_mcnemar_and_fishers_method():
    from src.stats_utils import mcnemar_test, fishers_method

    # Synthetic: two models with identical predictions -> maximal p-value scenario
    labels = [0, 1, 2, 3] * 25
    preds_a = labels.copy()
    preds_b = labels.copy()
    table, stat, p = mcnemar_test(preds_a, preds_b, labels)
    assert p > 0.9, f"Expected high p-value for identical predictions, got {p}"

    # Fisher's method: four consistent small p-values should combine to a smaller one
    stat, combined_p = fishers_method([0.03, 0.03, 0.03, 0.03])
    assert combined_p < 0.03, "Fisher's combined p-value should be smaller than individual p-values"

    print("test_mcnemar_and_fishers_method: PASSED")


if __name__ == "__main__":
    test_models_forward_pass()
    test_fft_magnitude_shape()
    test_mcnemar_and_fishers_method()
    print("\nAll sanity tests passed.")
