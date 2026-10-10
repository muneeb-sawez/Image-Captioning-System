"""Step 1: read Flickr8k captions, split BY IMAGE, and build the vocabulary.

Works with both common Flickr8k layouts:
  - Kaggle:   Images/ + captions.txt   (CSV with header image,caption)
  - Original: Flicker8k_Dataset/ + Flickr8k.token.txt   (image.jpg#0<TAB>caption)

Usage:
    python prepare_data.py --captions data/flickr8k/captions.txt --images data/flickr8k/Images
"""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict

from common import SPECIALS, tokenize, write_json


def read_captions(path):
    caps = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        first = f.readline()
        f.seek(0)
        if "\t" in first:  # original token file
            for line in f:
                if "\t" in line:
                    key, text = line.rstrip("\n").split("\t", 1)
                    caps[key.split("#")[0]].append(text)
        else:  # Kaggle captions.txt
            for row in csv.reader(f):
                if not row or row[0].strip().lower() == "image":
                    continue
                caps[row[0].strip()].append(",".join(row[1:]))
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/flickr8k/captions.txt")
    ap.add_argument("--images", default="data/flickr8k/Images")
    ap.add_argument("--work", default="work")
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    caps = read_captions(a.captions)
    names = sorted(n for n in caps if os.path.exists(os.path.join(a.images, n)))
    if not names:
        raise SystemExit("No images found. Check --captions and --images paths.")
    print(f"{len(names)} images with captions")

    random.Random(a.seed).shuffle(names)
    n_val = n_test = len(names) // 10
    split_names = {"test": names[:n_test], "val": names[n_test:n_test + n_val], "train": names[n_test + n_val:]}

    # Safety check: no image may appear in two splits (prevents data leakage)
    assert not (set(split_names["train"]) & set(split_names["val"]))
    assert not (set(split_names["train"]) & set(split_names["test"]))
    assert not (set(split_names["val"]) & set(split_names["test"]))
    print({k: len(v) for k, v in split_names.items()}, "- no overlap between splits")

    tokens = {s: {n: [tokenize(c) for c in caps[n]] for n in ns} for s, ns in split_names.items()}

    counts = Counter(w for lst in tokens["train"].values() for t in lst for w in t)  # vocab from TRAIN only
    itos = SPECIALS + sorted(w for w, c in counts.items() if c >= a.min_freq)
    print(f"Vocabulary size: {len(itos)} (words seen at least {a.min_freq} times)")

    write_json(os.path.join(a.work, "captions.json"), tokens)
    write_json(os.path.join(a.work, "vocab.json"), {"itos": itos})
    write_json(os.path.join(a.work, "meta.json"), {"images": a.images})
    print(f"Saved to {a.work}/")


if __name__ == "__main__":
    main()
