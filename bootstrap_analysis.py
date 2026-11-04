"""
Image-level paired bootstrap for the FADB-CGF revision (reviewer comment 3).
Input : results/<model>_seed<seed>_full.json  (keys: predictions, labels on the 1,213 filtered test images)
Output: model_ci.csv, per_class_recall_ci.csv, paired_diffs.csv, consistently_misclassified_glioma.csv
Usage : python bootstrap_analysis.py [results_dir] [out_dir]
"""
import json, sys, os, numpy as np, pandas as pd
from scipy import stats
R = (sys.argv[1] if len(sys.argv) > 1 else "results").rstrip("/") + "/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
MODELS = ["efficientnet_baseline","resnet50_baseline","fadb_cgf_concat","fadb_cgf_gated","frequency_only"]
SEEDS = [42,123,456,789]; K = 4; B = 10000
CLASSES = ["glioma","meningioma","no tumor","pituitary"]
rng = np.random.default_rng(2026)

P, L = {}, None
for m in MODELS:
    for s in SEEDS:
        d = json.load(open(f"{R}{m}_seed{s}_full.json"))
        P[(m,s)] = np.array(d["predictions"]); l = np.array(d["labels"])
        if L is None: L = l
        assert (L == l).all()
n = len(L); oh_t = np.eye(K)[L]

# stratified (by class) resampling weights: class sizes stay 393/297/123/400
W = np.zeros((B, n))
for c in range(K):
    idx = np.where(L == c)[0]
    draws = rng.choice(idx, size=(B, len(idx)))
    for b in range(B): W[b] += np.bincount(draws[b], minlength=n)

def metrics(w, pred):
    ok = (pred == L).astype(float); oh_p = np.eye(K)[pred]
    tp = (w * ok) @ oh_t; sup = w @ oh_t; pc = w @ oh_p
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    prec = np.divide(tp, pc, out=np.zeros_like(tp), where=pc > 0)
    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(tp), where=(prec+rec) > 0)
    return (w*ok).sum(1)/w.sum(1), rec.mean(1), f1.mean(1), rec          # acc, balanced acc, macro-F1, per-class recall

one = np.ones((1, n))
pt, bt = {}, {}                                       # point estimates, bootstrap draws (per seed)
for m in MODELS:
    for s in SEEDS:
        pt[(m,s)] = [x[0] if x.ndim == 1 else x[0] for x in metrics(one, P[(m,s)])]
        bt[(m,s)] = metrics(W, P[(m,s)])
KEYS = ["acc","bal","f1"]
def seedmean_point(m, i): return np.mean([pt[(m,s)][i] for s in SEEDS], axis=0)
def seedmean_boot(m, i):  return np.mean([bt[(m,s)][i] for s in SEEDS], axis=0)
def ci(x, a=95): return np.percentile(x, [(100-a)/2, 100-(100-a)/2], axis=0)

rows = []
for m in MODELS:
    for i, k in enumerate(KEYS):
        lo, hi = ci(seedmean_boot(m, i))
        rows.append(dict(model=m, metric=k, point=100*seedmean_point(m, i), lo95=100*lo, hi95=100*hi))
pd.DataFrame(rows).to_csv(f"{OUT}/model_ci.csv", index=False)

# per-class recall; exact Clopper-Pearson where every seed has recall = 100%
rows = []
for m in MODELS:
    pr = seedmean_point(m, 3); lo, hi = ci(seedmean_boot(m, 3))
    for c in range(K):
        nc = int((L == c).sum()); method = "bootstrap"
        if np.isclose(pr[c], 1.0):
            lo[c] = 0.025 ** (1/nc)  # exact Clopper-Pearson lower bound when all nc images are correct; hi[c] = 1.0; method = "Clopper-Pearson"
        rows.append(dict(model=m, cls=CLASSES[c], n=nc, recall=100*pr[c], lo95=100*lo[c], hi95=100*hi[c], method=method))
pd.DataFrame(rows).to_csv(f"{OUT}/per_class_recall_ci.csv", index=False)

# paired differences: (a) images only, seeds fixed; (b) images + seeds (same seed draw for both models)
COMPS = [("fadb_cgf_gated","efficientnet_baseline"),("fadb_cgf_concat","efficientnet_baseline"),
         ("fadb_cgf_gated","resnet50_baseline"),("fadb_cgf_concat","resnet50_baseline"),
         ("fadb_cgf_gated","fadb_cgf_concat"),("resnet50_baseline","efficientnet_baseline")]
S = rng.integers(0, len(SEEDS), (B, len(SEEDS)))
def combo(m, i):                                       # (B,) mean over drawn seeds, per resample
    arr = np.stack([bt[(m,s)][i] for s in SEEDS])      # (4,B)
    return arr[S, np.arange(B)[:, None]].mean(1)
rows = []
for a, b in COMPS:
    for i, k in enumerate(KEYS):
        pdiff = 100*(seedmean_point(a, i) - seedmean_point(b, i))
        d1 = 100*(seedmean_boot(a, i) - seedmean_boot(b, i)); d2 = 100*(combo(a, i) - combo(b, i))
        r = dict(a=a, b=b, metric=k, diff=pdiff)
        for tag, d in (("img", d1), ("imgseed", d2)):
            r[f"lo95_{tag}"], r[f"hi95_{tag}"] = ci(d, 95); r[f"lo90_{tag}"], r[f"hi90_{tag}"] = ci(d, 90)
            r[f"equiv1pp_{tag}"] = bool(r[f"lo90_{tag}"] > -1 and r[f"hi90_{tag}"] < 1)
        rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/paired_diffs.csv", index=False)

# images wrong in every run of the four full-capacity models
FULL = MODELS[:4]
wrong = np.all([P[(m,s)] != L for m in FULL for s in SEEDS], axis=0)
recs = []
for j in np.where(wrong)[0]:
    preds = {int(P[(m,s)][j]) for m in FULL for s in SEEDS}
    recs.append(dict(filtered_test_index=int(j), true=CLASSES[L[j]], predicted_classes="/".join(CLASSES[p] for p in sorted(preds)),
                     n_runs=16))
pd.DataFrame(recs).to_csv(f"{OUT}/consistently_misclassified_glioma.csv", index=False)
print("wrong in all 16 runs:", int(wrong.sum()), "by true class:", np.bincount(L[wrong], minlength=K).tolist())
print(pd.DataFrame(recs).groupby(["true","predicted_classes"]).size())
