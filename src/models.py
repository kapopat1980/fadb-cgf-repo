"""
Model definitions for the FADB-CGF experiment.

Includes:
  - SpatialBranch: EfficientNetB0 or ResNet50 backbone (ImageNet-pretrained)
  - FrequencyBranch: lightweight 4-block CNN over the FFT-magnitude spectrum
  - ConcatFusionHead / GatedFusionHead: the two fusion variants compared
  - FADB_CGF: full dual-branch model
  - FrequencyOnlyModel, BaselineModel: ablation and baseline models

See Section 3.3 of the manuscript for the architectural rationale.
"""
import torch
import torch.nn as nn
from torchvision import models


class SpatialBranch(nn.Module):
    """ImageNet-pretrained backbone, returns pooled feature embedding."""

    def __init__(self, backbone="efficientnet_b0", pretrained=True):
        super().__init__()
        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            self.features = net.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.out_dim = 1280
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.out_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


class FrequencyBranch(nn.Module):
    """Lightweight CNN over the FFT-magnitude spectrum (1-channel input)."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x).flatten(1)


class ConcatFusionHead(nn.Module):
    """Fusion variant A: concatenation."""

    def __init__(self, spatial_dim, freq_dim, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, spatial_feat, freq_feat):
        return self.fc(torch.cat([spatial_feat, freq_feat], dim=1))


class GatedFusionHead(nn.Module):
    """Fusion variant B: confidence-gated cross-attention.

    Projects both embeddings to a shared space, learns a per-sample scalar
    gate g in [0,1], and fuses as spatial_proj + g * freq_proj.
    """

    def __init__(self, spatial_dim, freq_dim, num_classes=4, proj_dim=256):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, proj_dim)
        self.freq_proj = nn.Linear(freq_dim, proj_dim)
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, 1), nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, spatial_feat, freq_feat):
        s = self.spatial_proj(spatial_feat)
        f = self.freq_proj(freq_feat)
        g = self.gate(torch.cat([s, f], dim=1))
        fused = s + g * f
        return self.fc(fused)


class FADB_CGF(nn.Module):
    """Frequency-Aware Dual-Branch Network with Confidence-Gated Fusion.

    fusion: "concat" or "gated" - selects which fusion head to use.
    """

    def __init__(self, backbone="efficientnet_b0", fusion="gated", num_classes=4, pretrained=True):
        super().__init__()
        self.spatial_branch = SpatialBranch(backbone, pretrained=pretrained)
        self.freq_branch = FrequencyBranch()
        if fusion == "concat":
            self.head = ConcatFusionHead(self.spatial_branch.out_dim, self.freq_branch.out_dim, num_classes)
        elif fusion == "gated":
            self.head = GatedFusionHead(self.spatial_branch.out_dim, self.freq_branch.out_dim, num_classes)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

    def forward(self, spatial_x, freq_x):
        s = self.spatial_branch(spatial_x)
        f = self.freq_branch(freq_x)
        return self.head(s, f)


class FrequencyOnlyModel(nn.Module):
    """Ablation: frequency branch alone, isolating whether it carries signal."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.freq_branch = FrequencyBranch()
        self.fc = nn.Sequential(nn.Linear(self.freq_branch.out_dim, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, spatial_x, freq_x):
        return self.fc(self.freq_branch(freq_x))


class BaselineModel(nn.Module):
    """Spatial-only baseline (EfficientNetB0 or ResNet50)."""

    def __init__(self, backbone="efficientnet_b0", num_classes=4, pretrained=True):
        super().__init__()
        self.spatial_branch = SpatialBranch(backbone, pretrained=pretrained)
        self.fc = nn.Linear(self.spatial_branch.out_dim, num_classes)

    def forward(self, spatial_x, freq_x=None):
        return self.fc(self.spatial_branch(spatial_x))
