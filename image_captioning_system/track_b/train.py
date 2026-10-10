"""Step 3: train the 2-layer GRU decoder with teacher forcing.

Saves the best checkpoint (lowest validation loss) to checkpoints/gru_best.pt in the format the
Flask website expects, and writes results/train_log.csv for the website's Models page.

Usage:
    python train.py
    python train.py --layers 1 --epochs 15      # for the 1-layer vs 2-layer ablation
"""
import argparse
import csv
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import END, PAD, START, UNK, read_json
from model import CaptionGRU


class CapData(Dataset):
    """One item per (image, caption) pair: (image feature, [<start>, w1, ..., <end>])."""

    def __init__(self, split_caps, index, feats, stoi, max_len):
        self.feats, self.stoi = feats, stoi
        self.items = [(index[img], [stoi.get(w, stoi[UNK]) for w in toks][:max_len])
                      for img, lst in split_caps.items() for toks in lst]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fi, ids = self.items[i]
        return self.feats[fi], torch.tensor([self.stoi[START]] + ids + [self.stoi[END]])


def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    caps = torch.zeros(len(batch), max(len(b[1]) for b in batch), dtype=torch.long)  # 0 = <pad>
    for i, (_, c) in enumerate(batch):
        caps[i, :len(c)] = c
    return feats, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=3, help="stop after this many epochs without improvement")
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="checkpoints/gru_best.pt")
    ap.add_argument("--log", default="results/train_log.csv")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    itos = read_json(os.path.join(a.work, "vocab.json"))["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    assert stoi[PAD] == 0
    caps = read_json(os.path.join(a.work, "captions.json"))
    fd = torch.load(os.path.join(a.work, "features.pt"))
    index = {n: i for i, n in enumerate(fd["names"])}

    loaders = {s: DataLoader(CapData(caps[s], index, fd["feats"], stoi, a.max_len), batch_size=a.batch,
                             shuffle=(s == "train"), collate_fn=collate) for s in ("train", "val")}
    print(f"train pairs: {len(loaders['train'].dataset)}, val pairs: {len(loaders['val'].dataset)}, device: {device}")

    cfg = {"embed_dim": a.embed, "hidden_dim": a.hidden, "num_layers": a.layers, "dropout": a.dropout}
    model = CaptionGRU(len(itos), **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    crit = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    def run(split, train):
        model.train(train)
        total, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for feats, c in loaders[split]:
                feats, c = feats.to(device), c.to(device)
                logits = model(feats, c)  # (B, T-1, V)
                loss = crit(logits.reshape(-1, logits.size(-1)), c[:, 1:].reshape(-1))
                if train:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                    opt.step()
                total += loss.item() * feats.size(0)
                n += feats.size(0)
        return total / n

    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    best, bad = float("inf"), 0
    with open(a.log, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_loss"])
        for epoch in range(1, a.epochs + 1):
            tr, va = run("train", True), run("val", False)
            log.writerow([epoch, round(tr, 4), round(va, 4)])
            f.flush()
            mark = ""
            if va < best:
                best, bad, mark = va, 0, "  <- best, saved"
                torch.save({"model_state": model.state_dict(), "itos": itos, "config": cfg}, a.ckpt)
            else:
                bad += 1
            print(f"epoch {epoch:2d}  train {tr:.4f}  val {va:.4f}{mark}", flush=True)
            if bad >= a.patience:
                print(f"Early stopping: no improvement for {a.patience} epochs.")
                break
    print(f"Best validation loss {best:.4f}. Checkpoint: {a.ckpt}")


if __name__ == "__main__":
    main()
