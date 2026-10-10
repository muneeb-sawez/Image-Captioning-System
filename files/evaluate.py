"""Step 4: score the trained model on the TEST split (BLEU) and optionally compare it with BLIP.

Writes results/metrics.json and results/comparison.csv in the formats the Flask website reads.

Usage:
    python evaluate.py                      # BLEU of the GRU against the human captions
    python evaluate.py --blip --limit 200   # also run BLIP on the first 200 test images
"""
import argparse
import csv
import os

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from common import END, START, ids_to_words, read_json, tokenize, write_json
from model import CaptionGRU

SMOOTH = SmoothingFunction().method1


def bleu_scores(refs, hyps):
    """BLEU-1 to BLEU-4. refs: per image, a list of token lists. hyps: per image, a token list."""
    out = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n if k < n else 0.0 for k in range(4))
        out[f"bleu{n}"] = round(corpus_bleu(refs, hyps, weights=weights, smoothing_function=SMOOTH), 4)
    return out


def agreement(blip_tokens, hyps):
    """BLEU-1 of our captions against BLIP's captions (BLIP used as a stand-in reference)."""
    return round(corpus_bleu([[b] for b in blip_tokens], hyps, weights=(1, 0, 0, 0), smoothing_function=SMOOTH), 4)


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
            out = model.generate(**inputs, num_beams=beams, max_new_tokens=40)
        caps.append(proc.decode(out[0], skip_special_tokens=True).strip())
        if i % 20 == 0:
            print(f"BLIP {i + 1}/{len(paths)}", flush=True)
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--ckpt", default="checkpoints/gru_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (0 = all)")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--blip", action="store_true", help="also caption with BLIP and compare")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device)
    itos = ck["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    model = CaptionGRU(len(itos), **ck["config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    fd = torch.load(os.path.join(a.work, "features.pt"))
    index = {n: i for i, n in enumerate(fd["names"])}
    caps = read_json(os.path.join(a.work, "captions.json"))[a.split]
    names = sorted(caps)[: a.limit or None]
    refs = [caps[n] for n in names]
    X = fd["feats"][[index[n] for n in names]].to(device)
    start, end = stoi[START], stoi[END]
    print(f"Evaluating on {len(names)} {a.split} images")

    greedy = []
    for i in range(0, len(names), 64):
        greedy += [ids_to_words(r, itos) for r in model.generate(X[i:i + 64], start, end).tolist()]
    beam = [ids_to_words(model.beam_search(X[i:i + 1], start, end, a.beam), itos) for i in range(len(names))]

    result = {"gru_greedy": bleu_scores(refs, greedy), "gru_beam": bleu_scores(refs, beam)}
    blip_caps = [""] * len(names)
    if a.blip:
        root = read_json(os.path.join(a.work, "meta.json"))["images"]
        blip_caps = run_blip([os.path.join(root, n) for n in names])
        blip_tok = [tokenize(c) for c in blip_caps]
        result["blip"] = bleu_scores(refs, blip_tok)
        result["gru_greedy"]["agreement"] = agreement(blip_tok, greedy)
        result["gru_beam"]["agreement"] = agreement(blip_tok, beam)

    path = os.path.join(a.results, "metrics.json")
    metrics = read_json(path) if os.path.exists(path) else {}
    for key, val in result.items():
        metrics[key] = {**metrics.get(key, {}), **val}  # keeps any 'manual' ratings you added
    write_json(path, metrics, indent=2)

    with open(os.path.join(a.results, "comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "human", "blip", "gru"])
        for n, r, b, g in zip(names, refs, blip_caps, beam):
            w.writerow([n, " ".join(r[0]), b, " ".join(g)])

    for key, val in result.items():
        print(key, val)
    print(f"Saved {path} and {a.results}/comparison.csv")


if __name__ == "__main__":
    main()
