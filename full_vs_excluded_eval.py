"""
Comment 2: evaluate saved checkpoints on the FULL 1,600-image test set, then split into
excluded (387) and retained (1,213) subsets.  TEMPLATE: fill the three TODO blocks from your repo.
Output: preds/<model>_seed<seed>.csv  and  full_excluded_retained.csv
"""
import json, os, numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

CKPT_DIR   = "checkpoints"                      # TODO: folder with <model>_seed<seed>.pt (weights-only is enough)
TEST_DIR   = "brain-tumor-mri-dataset/Testing"  # Kaggle v2 test folder
LEAK_JSON  = "results/leakage_ssim_verified.json"
MODELS = ["efficientnet_baseline","resnet50_baseline","fadb_cgf_concat","fadb_cgf_gated","frequency_only"]
SEEDS  = [42,123,456,789]
dev = "cuda" if torch.cuda.is_available() else "cpu"

# TODO 1: import YOUR model builders and test dataset (identical preprocessing to training:
#         224x224, ImageNet norm, NO augmentation, FFT input from un-augmented grayscale image)
# from your_repo.models import build_model
# from your_repo.data  import MRIDataset          # must return (inputs, label); keep sorted file order
# test_ds = MRIDataset(TEST_DIR, train=False); paths = test_ds.paths   # full paths, 1,600 items

def rel(p):                       # normalise to "class/filename" so Kaggle and local paths match
    q = p.replace("\\", "/").split("/"); return f"{q[-2]}/{q[-1]}"

leak = {rel(x[0]) for x in json.load(open(LEAK_JSON))["verified_leaks"]}
excluded = np.array([rel(p) in leak for p in paths]); assert excluded.sum() == 387

os.makedirs("preds", exist_ok=True); rows = []
loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
for m in MODELS:
    for s in SEEDS:
        net = build_model(m).to(dev)                                    # TODO 2: your constructor
        sd = torch.load(f"{CKPT_DIR}/{m}_seed{s}.pt", map_location=dev) # TODO 3: key if full checkpoint, e.g. sd=sd["model_state_dict"]
        net.load_state_dict(sd); net.eval(); pred = []
        with torch.no_grad():
            for x, _ in loader:
                x = [t.to(dev) for t in x] if isinstance(x,(list,tuple)) else x.to(dev)
                pred.append(net(*x).argmax(1).cpu() if isinstance(x,list) else net(x).argmax(1).cpu())
        pred = torch.cat(pred).numpy(); y = np.array(test_ds.labels)
        pd.DataFrame({"path":[rel(p) for p in paths],"label":y,"pred":pred,"excluded":excluded}).to_csv(f"preds/{m}_seed{s}.csv",index=False)
        for name, mask in [("full",np.ones_like(excluded)),("excluded",excluded),("retained",~excluded)]:
            rows.append(dict(model=m,seed=s,subset=name,n=int(mask.sum()),
                acc=accuracy_score(y[mask],pred[mask]), bal_acc=balanced_accuracy_score(y[mask],pred[mask]),
                macro_f1=f1_score(y[mask],pred[mask],average="macro")))
df = pd.DataFrame(rows); df.to_csv("full_excluded_retained_per_run.csv", index=False)
summ = df.groupby(["model","subset"])[["acc","bal_acc","macro_f1"]].agg(["mean","std"]).round(4)
summ.to_csv("full_excluded_retained.csv"); print(summ)

# SANITY CHECKS (must pass before using any number):
ret = df[(df.subset=="retained")&(df.model=="efficientnet_baseline")].sort_values("seed").acc.round(4).tolist()
print("retained-set accuracy, EfficientNetB0 per seed (paper Table 6: .9431 .9431 .9406 .9489):", ret)
full = df[(df.subset=="full")&(df.model=="efficientnet_baseline")].acc.mean()
print("full-set accuracy EfficientNetB0:", round(full,4), "(must be <= 0.9575, the theoretical ceiling)")
