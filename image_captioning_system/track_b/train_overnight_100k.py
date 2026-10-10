"""Overnight Large-Scale Training Pipeline (100,000 Images).
Trains VisionEncoderDecoder on MS-COCO 2017 + Flickr8k + Custom Portraits.
Optimized for continuous overnight execution with checkpoint safety and thermal protection.

Usage:
    python train_overnight_100k.py --epochs 5 --batch 8 --lr 3e-5
"""
import argparse
import csv
import json
import os
import random
import time
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class COCODataset(Dataset):
    def __init__(self, items, processor, tokenizer, max_len=30):
        self.items = items
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        pixel_values = self.processor(images=[img], return_tensors="pt").pixel_values[0]
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        labels = tokens.input_ids[0]
        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in labels]
        return pixel_values, torch.tensor(labels, dtype=torch.long)


def load_dataset_pairs(coco_ann_path, coco_img_dir, target_count=100000):
    pairs = []
    
    # 1. Load MS-COCO if available
    if os.path.exists(coco_ann_path) and os.path.exists(coco_img_dir):
        print(f"[Data] Parsing MS-COCO annotations from {coco_ann_path}...")
        with open(coco_ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id in id_to_file:
                img_path = os.path.join(coco_img_dir, id_to_file[img_id])
                if os.path.exists(img_path):
                    pairs.append((img_path, ann["caption"].strip()))
                    if len(pairs) >= target_count:
                        break
        print(f"[Data] Loaded {len(pairs)} pairs from MS-COCO.")

    # 2. Add Flickr8k & Custom Photos
    flickr_ann = os.path.join(BASE_DIR, "data", "flickr8k", "captions.txt")
    flickr_dir = os.path.join(BASE_DIR, "data", "flickr8k", "Images")
    if os.path.exists(flickr_ann):
        with open(flickr_ann, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2 and parts[0] != "image":
                    p = os.path.join(flickr_dir, parts[0])
                    if os.path.exists(p):
                        # Give custom user photos extra frequency weight
                        weight = 5 if "muneeb_" in parts[0] else 1
                        for _ in range(weight):
                            pairs.append((p, parts[1].strip()))

    print(f"[Data] Total training dataset assembled: {len(pairs)} image-caption pairs.")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_ann", default=os.path.join(BASE_DIR, "data", "coco", "annotations", "captions_train2017.json"))
    parser.add_argument("--coco_img", default=os.path.join(BASE_DIR, "data", "coco", "train2017"))
    parser.add_argument("--target_count", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--save_dir", default=os.path.join(BASE_DIR, "checkpoints", "overnight_100k_best"))
    parser.add_argument("--webapp_save_dir", default=os.path.join(BASE_DIR, "webapp", "checkpoints", "vit_gpt2_finetuned"))
    parser.add_argument("--log_file", default=os.path.join(BASE_DIR, "results", "overnight_100k_log.csv"))
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Starting Overnight 100K Trainer on device: {dev}")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(dev)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    pairs = load_dataset_pairs(args.coco_ann, args.coco_img, args.target_count)
    if not pairs:
        print("[Error] No training data found! Please run download_coco.py first.")
        return

    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(0.95 * len(pairs))
    train_data = pairs[:split_idx]
    val_data = pairs[split_idx:]

    train_loader = DataLoader(COCODataset(train_data, processor, tokenizer), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(COCODataset(val_data, processor, tokenizer), batch_size=args.batch, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.webapp_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    log_rows = [["epoch", "train_loss", "val_loss", "epoch_duration_sec"]]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for i, (pixels, labels) in enumerate(train_loader):
            pixels, labels = pixels.to(dev), labels.to(dev)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                outputs = model(pixel_values=pixels, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if (i + 1) % 200 == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pixels, labels in val_loader:
                pixels, labels = pixels.to(dev), labels.to(dev)
                with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                    out = model(pixel_values=pixels, labels=labels)
                    val_loss += out.loss.item()
        avg_val = val_loss / len(val_loader)
        duration = round(time.time() - t0, 1)

        print(f"\n--> [Epoch {epoch} Done] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Time: {duration}s")
        log_rows.append([epoch, round(avg_train, 4), round(avg_val, 4), duration])

        # Save checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            processor.save_pretrained(args.save_dir)
            # Sync to webapp
            model.save_pretrained(args.webapp_save_dir)
            tokenizer.save_pretrained(args.webapp_save_dir)
            processor.save_pretrained(args.webapp_save_dir)
            print(f"    [Saved] Best model checkpoint saved to {args.save_dir} and synced to webapp!")

        # 15s thermal cooldown for laptop GPU health
        time.sleep(15)

    with open(args.log_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(log_rows)

    print("\n[Complete] Overnight 100K training finished successfully!")

if __name__ == "__main__":
    main()
