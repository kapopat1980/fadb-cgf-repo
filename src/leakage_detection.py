"""
Two-stage train-test leakage detection pipeline.

Stage 1: perceptual hash (pHash) proximity, Hamming distance <= threshold,
         identifies CANDIDATE near-duplicates (fast, but prone to false
         positives on low-texture images).
Stage 2: pixel-level SSIM verification confirms or rejects each candidate.

Only pairs confirmed by BOTH stages are treated as genuine leakage. See
Section 3.2 and Figure 1 of the manuscript for the full rationale -
pHash alone produced a high false-positive rate on the low-texture
"no tumor" class in this dataset.
"""
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import imagehash
except ImportError:
    imagehash = None  # only required if running the hashing stage fresh

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None  # only required if running SSIM verification fresh

from src.config import CLASSES, HAMMING_THRESHOLD, SSIM_THRESHOLD

POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def hash_folder(root):
    """Computes a pHash for every image in each class subfolder of `root`."""
    if imagehash is None:
        raise ImportError("pip install imagehash to run hashing")
    records = []
    for c in CLASSES:
        for f in sorted((Path(root) / c).glob("*")):
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            h = imagehash.phash(Image.open(f).convert("L"))
            records.append((str(f), c, str(h)))
    return records


def hex_hashes_to_uint8_matrix(hex_list):
    ints = np.array([int(h, 16) for h in hex_list], dtype=np.uint64)
    return ints.view(np.uint8).reshape(-1, 8)


def pairwise_hamming(train_bytes, test_bytes):
    """Returns (n_test, n_train) Hamming distance matrix."""
    xor = train_bytes[None, :, :] ^ test_bytes[:, None, :]
    dist = POPCOUNT_TABLE[xor].sum(axis=2)
    return dist


def find_phash_candidates(train_records, test_records, threshold=HAMMING_THRESHOLD):
    """Stage 1: returns list of [test_path, class, matched_train_path, hamming_dist]."""
    train_bytes = hex_hashes_to_uint8_matrix([r[2] for r in train_records])
    test_bytes = hex_hashes_to_uint8_matrix([r[2] for r in test_records])
    dist_matrix = pairwise_hamming(train_bytes, test_bytes)
    min_dist = dist_matrix.min(axis=1)
    argmin_train = dist_matrix.argmin(axis=1)

    candidates = []
    for i, (test_path, test_class, _) in enumerate(test_records):
        if min_dist[i] <= threshold:
            train_path, _, _ = train_records[argmin_train[i]]
            candidates.append([test_path, test_class, train_path, int(min_dist[i])])
    return candidates


def load_gray_array(path, size=128):
    img = Image.open(path).convert("L").resize((size, size))
    return np.array(img, dtype=np.float64) / 255.0


def verify_with_ssim(candidates, threshold=SSIM_THRESHOLD):
    """Stage 2: filters pHash candidates by pixel-level SSIM.

    Returns (verified_leaks, rejected_false_positives), where each entry is
    [test_path, class, train_path, hamming_dist, ssim_score].
    """
    if ssim is None:
        raise ImportError("pip install scikit-image to run SSIM verification")
    verified, rejected = [], []
    for test_path, cls, train_path, dist in candidates:
        a = load_gray_array(test_path)
        b = load_gray_array(train_path)
        score = ssim(a, b, data_range=1.0)
        entry = [test_path, cls, train_path, dist, float(score)]
        (verified if score >= threshold else rejected).append(entry)
    return verified, rejected


def run_leakage_pipeline(train_dir, test_dir):
    """End-to-end: hash both splits, find candidates, verify with SSIM.

    Returns (verified_leaks, rejected_false_positives, n_test_total).
    """
    train_records = hash_folder(train_dir)
    test_records = hash_folder(test_dir)
    candidates = find_phash_candidates(train_records, test_records)
    verified, rejected = verify_with_ssim(candidates)
    return verified, rejected, len(test_records)
