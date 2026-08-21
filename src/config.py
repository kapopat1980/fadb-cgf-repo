"""
Configuration for the FADB-CGF brain tumor MRI classification experiment.

Adjust DATA_ROOT to point at your local copy of the Masoud Nickparvar
"Brain Tumor MRI Dataset" (balanced/augmented 7,200-image variant used in
this study). See README.md for dataset acquisition instructions.
"""
from pathlib import Path

# ---- Paths (override via environment or edit directly) ----
DATA_ROOT = Path("./data/brain-tumor-mri-dataset")
TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Testing"

WORK_DIR = Path("./outputs")
CKPT_DIR = WORK_DIR / "checkpoints"
RESULTS_DIR = WORK_DIR / "results"

# ---- Dataset ----
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224

# ---- Training hyperparameters ----
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEEDS = [42, 123, 456, 789]

# ---- Leakage detection ----
HAMMING_THRESHOLD = 5   # out of 64 bits
SSIM_THRESHOLD = 0.90

# ---- Statistical testing ----
ALPHA = 0.05

# ---- Model registry: maps model name -> constructor closure ----
def get_model_registry():
    """Deferred import to avoid pulling in torch when this module is used
    purely for path/constant access (e.g., in analysis scripts)."""
    from src.models import BaselineModel, FrequencyOnlyModel, FADB_CGF
    return {
        "efficientnet_baseline": lambda: BaselineModel("efficientnet_b0"),
        "resnet50_baseline": lambda: BaselineModel("resnet50"),
        "frequency_only": lambda: FrequencyOnlyModel(),
        "fadb_cgf_concat": lambda: FADB_CGF("efficientnet_b0", fusion="concat"),
        "fadb_cgf_gated": lambda: FADB_CGF("efficientnet_b0", fusion="gated"),
    }
