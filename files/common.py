"""Shared helpers for Part B (no PyTorch needed here)."""
import json
import os
import re

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIALS = [PAD, START, END, UNK]  # PAD must stay at index 0


def tokenize(text):
    """Lowercase and keep only letter sequences, e.g. 'A dog, running!' -> ['a','dog','running']."""
    return re.findall(r"[a-z]+", text.lower())


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj, indent=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def ids_to_words(ids, itos):
    """Turn a list of word ids into words, stopping at <end>."""
    words = []
    for i in ids:
        w = itos[i]
        if w == END:
            break
        if w not in (PAD, START, UNK):
            words.append(w)
    return words
