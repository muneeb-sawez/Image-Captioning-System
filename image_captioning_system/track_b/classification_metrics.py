"""Numerical accuracy / precision / recall / specificity / F1 for the caption models.

Run from your track_b folder (it needs work/, checkpoints/ and model.py as produced by the training steps):
    python classification_metrics.py                    # GRU only (fast)
    python classification_metrics.py --blip --limit 200 # also BLIP on the first 200 test images (slow on CPU)
    python classification_metrics.py --blip             # BLIP on the whole test split

Two kinds of numbers are produced:
  1. Word-level classification metrics (see word_metrics.py for exactly how TP/FP/FN/TN are defined),
     once over ALL vocabulary words and once over CONTENT words only (stopwords like "a", "the", "in" removed).
  2. Token-level metrics for the GRU under teacher forcing on the test split: next-word accuracy
     (top-1 and top-5), loss, and perplexity.
Results are printed and saved to results/classification_metrics.json
"""
import argparse
import json
import math
import os
import re
import sys

# Ensure local imports work regardless of current working directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from model import CaptionGRU
from word_metrics import STOPWORDS, evaluate_sets

SPECIALS = {"<pad>", "<start>", "<end>", "<unk>"}


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def as_tokens(c):
    return c if isinstance(c, list) else tokenize(c)


def to_words(ids, itos):
    out = []
    for i in ids:
        w = itos[i]
        if w == "<end>":
            break
        if w not in SPECIALS:
            out.append(w)
    return out


def token_level(model, caps_split, index, feats, stoi, device, max_len=20):
    """Teacher-forced next-word accuracy, loss and perplexity over every (image, caption) pair."""
    items = []
    for img, lst in caps_split.items():
        for c in lst:
            ids = [stoi["<start>"]] + [stoi.get(w, stoi["<unk>"]) for w in as_tokens(c)[:max_len]] + [stoi["<end>"]]
            items.append((index[img], ids))
    total_loss, c1, c5, n = 0.0, 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(items), 128):
            chunk = items[i:i + 128]
            caps = torch.zeros(len(chunk), max(len(x[1]) for x in chunk), dtype=torch.long)
            for j, (_, ids) in enumerate(chunk):
                caps[j, :len(ids)] = torch.tensor(ids)
            caps = caps.to(device)
            fb = feats[[x[0] for x in chunk]].to(device)
            logits = model(fb, caps)
            target = caps[:, 1:]
            mask = target != 0
            ce = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                             reduction="none").view_as(target)
            top5 = logits.topk(5, dim=-1).indices
            total_loss += ce[mask].sum().item()
            c1 += ((top5[..., 0] == target) & mask).sum().item()
            c5 += ((top5 == target.unsqueeze(-1)).any(-1) & mask).sum().item()
            n += mask.sum().item()
    return {"token_accuracy_top1": c1 / n, "token_accuracy_top5": c5 / n,
            "loss": total_loss / n, "perplexity": math.exp(total_loss / n), "tokens": n}


def run_blip(paths, beams=5):
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor
    model_id = "Salesforce/blip-image-captioning-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    caps = []
    for i, p in enumerate(paths):
        inputs = proc(images=Image.open(p).convert("RGB"), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                num_beams=beams,
                max_new_tokens=40,
                min_length=6,
                repetition_penalty=1.25,
                no_repeat_ngram_size=2
            )
        caps.append(proc.decode(out[0], skip_special_tokens=True).strip())
        if (i + 1) % 20 == 0 or (i + 1) == len(paths):
            print(f"BLIP {i + 1}/{len(paths)}", flush=True)
    return caps


def show(title, table):
    print(f"\n{title}")
    print(f"{'Model':<16}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'Specificity':>13}{'F1':>9}")
    for name, s in table.items():
        print(f"{name:<16}" + f"{100 * s['accuracy']:>9.2f}%{100 * s['precision']:>10.2f}%"
              f"{100 * s['recall']:>8.2f}%{100 * s['specificity']:>12.2f}%{100 * s['f1']:>8.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--ckpt", default="checkpoints/gru_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (0 = all)")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--blip", action="store_true")
    ap.add_argument("--images", default=None, help="image folder (only needed with --blip if work/meta.json lacks it)")
    ap.add_argument("--out", default="results/classification_metrics.json")
    a = ap.parse_args()

    # Intelligent path discovery if executed from track_b directory
    if not os.path.exists(a.work) and os.path.exists(os.path.join("..", a.work)):
        a.work = os.path.join("..", a.work)
    if not os.path.exists(a.ckpt) and os.path.exists(os.path.join("..", a.ckpt)):
        a.ckpt = os.path.join("..", a.ckpt)
    if not os.path.isabs(a.out) and not os.path.exists(os.path.dirname(a.out) or ".") and os.path.exists(".."):
        parent_out_dir = os.path.join("..", os.path.dirname(a.out))
        if os.path.exists(parent_out_dir):
            a.out = os.path.join("..", a.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device)
    itos = ck["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    model = CaptionGRU(len(itos), **ck["config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    fd = torch.load(os.path.join(a.work, "features.pt"))
    index = {n: i for i, n in enumerate(fd["names"])}
    caps_split = read_json(os.path.join(a.work, "captions.json"))[a.split]
    names = sorted(caps_split)[: a.limit or None]
    truth = [[w for c in caps_split[n] for w in as_tokens(c)] for n in names]  # all words from all 5 captions
    X = fd["feats"][[index[n] for n in names]].to(device)
    start, end = stoi["<start>"], stoi["<end>"]
    universe = set(itos) - SPECIALS
    print(f"{len(names)} {a.split} images, vocabulary of {len(universe)} words")

    preds = {"GRU greedy": [], "GRU beam": []}
    for i in range(0, len(names), 64):
        preds["GRU greedy"] += [to_words(r, itos) for r in model.generate(X[i:i + 64], start, end).tolist()]
    preds["GRU beam"] = [to_words(model.beam_search(X[i:i + 1], start, end, a.beam), itos) for i in range(len(names))]

    if a.blip:
        meta_path = os.path.join(a.work, "meta.json")
        root = a.images or (read_json(meta_path).get("images") if os.path.exists(meta_path) else None)
        if root and os.path.exists(root):
            preds["BLIP"] = [tokenize(c) for c in run_blip([os.path.join(root, n) for n in names])]
        else:
            print("Notice: Flickr8k image folder not found; skipping BLIP word metrics (GRU metrics running normally).")

    all_words = {m: evaluate_sets(p, truth, universe) for m, p in preds.items()}
    content = {m: evaluate_sets(p, truth, universe, drop=STOPWORDS) for m, p in preds.items()}
    tok = token_level(model, caps_split, index, fd["feats"], stoi, device)

    show("WORD-LEVEL METRICS, ALL VOCABULARY WORDS", all_words)
    show("WORD-LEVEL METRICS, CONTENT WORDS ONLY (stopwords removed)", content)
    print("\nTOKEN-LEVEL (GRU, teacher forcing, whole test split)")
    print(f"  next-word accuracy top-1: {100 * tok['token_accuracy_top1']:.2f}%")
    print(f"  next-word accuracy top-5: {100 * tok['token_accuracy_top5']:.2f}%")
    print(f"  loss: {tok['loss']:.4f}   perplexity: {tok['perplexity']:.2f}   ({tok['tokens']} tokens)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"images": len(names), "vocab_words": len(universe), "all_words": all_words,
                   "content_words": content, "token_level": tok}, f, indent=2)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
