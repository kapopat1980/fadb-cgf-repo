"""
Training loop with per-epoch checkpointing and automatic resumption.

Designed for interrupted/multi-session environments (e.g., Kaggle's 12-hour
session limit) - re-running train_one_run() on a partially-trained model
resumes from the last saved epoch rather than restarting.
"""
import torch
import torch.nn as nn

from src.config import CKPT_DIR, EPOCHS, LR, WEIGHT_DECAY
from src.data import make_loaders


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_run(model_fn, model_name, seed, train_dataset, test_dataset,
                   device, epochs=EPOCHS, lr=LR):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{model_name}_seed{seed}.pt"
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(train_dataset, test_dataset, seed)

    model = model_fn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    best_val_acc = 0.0
    history = []

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history = ckpt.get("history", [])
        if start_epoch >= epochs:
            return model, history

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for spatial_x, freq_x, labels in train_loader:
            spatial_x, freq_x, labels = spatial_x.to(device), freq_x.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(spatial_x, freq_x)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(spatial_x, freq_x)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for spatial_x, freq_x, labels in val_loader:
                spatial_x, freq_x, labels = spatial_x.to(device), freq_x.to(device), labels.to(device)
                outputs = model(spatial_x, freq_x)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"[{model_name} seed{seed}] epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(), "sched_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc, "history": history,
        }, ckpt_path)

    return model, history
