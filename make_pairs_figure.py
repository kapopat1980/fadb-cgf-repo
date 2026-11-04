"""
Fig. 5 for the revision (reviewer comment 1): illustrative accepted and rejected pHash/SSIM pairs,
chosen by PRE-DEFINED, seeded rules (not by visual preference). Illustration only - not a validation.
Usage: python make_pairs_figure.py --json leakage_ssim_verified.json --root /path/to/brain-tumor-mri-dataset --out fig5_pairs.png
"""
import argparse, json, random, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--json", default="leakage_ssim_verified.json"); ap.add_argument("--root", required=True)
ap.add_argument("--out", default="fig5_pairs.png"); ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args(); rnd = random.Random(a.seed)
d = json.load(open(a.json)); acc, rej = d["verified_leaks"], d["rejected_false_positives"]
lab = lambda p: p.replace("\\", "/").split("/")[-2]
loc = lambda p: f"{a.root}/" + "/".join(p.replace("\\", "/").split("/")[-3:])
used = set()
def pick(pool, k, key=None, reverse=False):
    pool = [x for x in pool if x[0] not in used]
    pool = sorted(pool, key=key, reverse=reverse) if key else rnd.sample(pool, min(k, len(pool)))
    out = pool[:k]; used.update(x[0] for x in out); return out
groups = [
 ("Accepted (no tumor)",           pick([x for x in acc if x[1] == "notumor"], 2)),
 ("Accepted (meningioma)",         pick([x for x in acc if x[1] == "meningioma"], 2)),
 ("Accepted, lowest SSIM",         pick(acc, 2, key=lambda x: x[4])),
 ("Accepted, label conflict",      pick([x for x in acc if x[1] != lab(x[2])], 2)),
 ("Rejected, highest SSIM",        pick(rej, 2, key=lambda x: x[4], reverse=True)),
 ("Rejected, identical hash",      pick([x for x in rej if x[3] == 0], 2)),
]
pairs = [(g, x) for g, xs in groups for x in xs]
ncol = 4; nrow = int(np.ceil(len(pairs) / ncol))
fig, axs = plt.subplots(nrow, ncol * 2, figsize=(ncol * 4.2, nrow * 2.9))
for ax in axs.ravel(): ax.axis("off")
for i, (g, x) in enumerate(pairs):
    r, c = divmod(i, ncol)
    for j, (p, t) in enumerate([(x[0], "Test"), (x[2], "Train")]):
        ax = axs[r, 2 * c + j]; ax.imshow(Image.open(loc(p)).convert("L"), cmap="gray"); ax.axis("off")
        ax.set_title(f"{t}: {lab(p)}", fontsize=7)
    axs[r, 2 * c].text(0.0, -0.08, f"{g}\nHamming {x[3]}, SSIM {x[4]:.3f}", transform=axs[r, 2 * c].transAxes, fontsize=7, va="top")
plt.tight_layout(h_pad=2.2); plt.savefig(a.out, dpi=300); print("saved", a.out, len(pairs), "pairs")
