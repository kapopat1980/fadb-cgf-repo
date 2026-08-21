"""
Dataset and preprocessing for the FADB-CGF experiment.

BrainTumorDataset returns (spatial_tensor, freq_tensor, label) per sample,
where freq_tensor is the log-magnitude 2D FFT spectrum of the grayscale image.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from src.config import CLASSES, CLASS_TO_IDX, IMG_SIZE, BATCH_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def compute_fft_magnitude(pil_img, size=IMG_SIZE):
    """Log-magnitude 2D FFT spectrum, returned as a single-channel tensor."""
    img = pil_img.convert("L").resize((size, size))
    arr = np.array(img, dtype=np.float32)
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return torch.from_numpy(magnitude).unsqueeze(0).float()


class BrainTumorDataset(Dataset):
    """Returns (spatial_tensor, freq_tensor, label). Excludes any leaked
    files if `exclude_files` is provided (see src/leakage_detection.py)."""

    def __init__(self, root, transform, exclude_files=None):
        self.samples = []
        exclude_files = set(exclude_files or [])
        for c in CLASSES:
            for f in sorted((Path(root) / c).glob("*")):
                if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                if str(f) in exclude_files:
                    continue
                self.samples.append((f, CLASS_TO_IDX[c]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        spatial = self.transform(img)
        freq = compute_fft_magnitude(img)
        return spatial, freq, label


def make_loaders(train_dataset, test_dataset, seed, val_frac=0.15, batch_size=BATCH_SIZE):
    """Splits train_dataset into train/val (seeded) and wraps all three in DataLoaders."""
    n_val = int(len(train_dataset) * val_frac)
    n_train = len(train_dataset) - n_val
    train_sub, val_sub = random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader
