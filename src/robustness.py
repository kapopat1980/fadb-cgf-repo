"""
Robustness evaluation under image perturbation: Gaussian noise, blur, and
brightness shift. This is the evaluation identified as incomplete in the
manuscript's Limitations (Section 5.5) at time of submission - included
here so it can be run and the results folded into the paper before final
publication.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score


def add_gaussian_noise(img_tensor, std=0.08):
    return img_tensor + torch.randn_like(img_tensor) * std


def apply_perturbation(spatial_x, kind):
    if kind == "noise":
        return add_gaussian_noise(spatial_x, std=0.08)
    elif kind == "blur":
        return TF.gaussian_blur(spatial_x, kernel_size=[5, 5], sigma=[2.0, 2.0])
    elif kind == "brightness":
        return torch.clamp(spatial_x * 1.3, spatial_x.min(), spatial_x.max())
    else:
        return spatial_x


def evaluate_robustness(model, test_loader, device, kinds=("clean", "noise", "blur", "brightness")):
    model.eval()
    results = {}
    for kind in kinds:
        all_preds, all_labels = [], []
        with torch.no_grad():
            for spatial_x, freq_x, labels in test_loader:
                spatial_x, freq_x = spatial_x.to(device), freq_x.to(device)
                if kind != "clean":
                    spatial_x = apply_perturbation(spatial_x, kind)
                outputs = model(spatial_x, freq_x)
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())
        results[kind] = accuracy_score(all_labels, all_preds)
    return results
