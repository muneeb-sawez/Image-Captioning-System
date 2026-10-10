"""Part A: image captioning with a pre-trained model (BLIP).

Captions every image in a folder and saves a CSV with empty columns
for your manual ratings.

Usage:
    python part_a.py test_images --beams 5 --out part_a_results.csv
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID = "Salesforce/blip-image-captioning-base"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Caption a folder of images with BLIP.")
    ap.add_argument("folder", help="folder containing the test images")
    ap.add_argument("--beams", type=int, default=5, help="beam size (1 = greedy)")
    ap.add_argument("--prompt", default=None, help='optional caption start, e.g. "a photo of"')
    ap.add_argument("--out", default="part_a_results.csv", help="output CSV file")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found in {args.folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} (the first run downloads about 1 GB)...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()

    rows = []
    for path in files:
        image = Image.open(path).convert("RGB")
        start = time.time()
        inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                num_beams=args.beams,
                max_new_tokens=40,
                min_length=6,
                repetition_penalty=1.25,
                length_penalty=1.0,
                no_repeat_ngram_size=2
            )

        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        rows.append({"image": path.name, "caption": caption,
                     "seconds": round(time.time() - start, 2),
                     "accuracy": "", "fluency": "", "remarks": ""})
        print(f"{path.name}: {caption}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption", "seconds", "accuracy", "fluency", "remarks"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} captions to {args.out}")


if __name__ == "__main__":
    main()
