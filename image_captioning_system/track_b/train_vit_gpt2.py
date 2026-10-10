"""High-Performance Fine-Tuning Pipeline for VisionEncoderDecoder (ViT + GPT-2).
Optimized for NVIDIA RTX 4050 GPU (FP16 Mixed Precision, Tensor Cores).

Saves the fine-tuned model checkpoint directly to:
    checkpoints/vit_gpt2_finetuned and webapp/checkpoints/vit_gpt2_finetuned
"""
import argparse
import csv
import os
import random
import time
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

class CaptionDataset(Dataset):
    def __init__(self, img_dir, pairs, processor, tokenizer, max_len=30):
        self.img_dir = img_dir
        self.pairs = pairs
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, caption = self.pairs[idx]
        img_path = os.path.join(self.img_dir, img_name)
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
        # Replace padding token id with -100 to ignore in cross-entropy loss
        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in labels]
        return pixel_values, torch.tensor(labels, dtype=torch.long)


def load_pairs(captions_file):
    pairs = []
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            parts = line.split(",", 1)
            img_name = parts[0].strip()
            caption = parts[1].strip()
            if img_name != "image":
                pairs.append((img_name, caption))
    return pairs


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=os.path.join(BASE_DIR, "data", "flickr8k", "Images"))
    parser.add_argument("--captions", default=os.path.join(BASE_DIR, "data", "flickr8k", "captions.txt"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--pairs_count", type=int, default=3000, help="Number of balanced pairs to fine-tune on")
    parser.add_argument("--save_dir", default=os.path.join(BASE_DIR, "checkpoints", "vit_gpt2_finetuned"))
    parser.add_argument("--webapp_save_dir", default=os.path.join(BASE_DIR, "webapp", "checkpoints", "vit_gpt2_finetuned"))
    parser.add_argument("--log_file", default=os.path.join(BASE_DIR, "results", "vit_gpt2_train_log.csv"))
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Training ViT-GPT2 on device: {dev} (GPU: {torch.cuda.get_device_name(0) if dev=='cuda' else 'CPU'})")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(dev)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    # Load and balance dataset pairs
    all_pairs = load_pairs(args.captions)
    print(f"[Data] Total available pairs: {len(all_pairs)}")

    random.seed(42)
    # Ensure custom photos are always included in training
    custom_pairs = [p for p in all_pairs if "muneeb_" in p[0]]
    standard_pairs = [p for p in all_pairs if "muneeb_" not in p[0]]
    selected_sample = custom_pairs * 5 + random.sample(standard_pairs, min(args.pairs_count, len(standard_pairs)))

    split_idx = int(0.9 * len(selected_sample))
    train_data = selected_sample[:split_idx]
    val_data = selected_sample[split_idx:]
    print(f"[Data] Training pairs: {len(train_data)} | Validation pairs: {len(val_data)}")

    train_loader = DataLoader(
        CaptionDataset(args.img_dir, train_data, processor, tokenizer),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True if dev == "cuda" else False
    )
    val_loader = DataLoader(
        CaptionDataset(args.img_dir, val_data, processor, tokenizer),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.webapp_save_dir, exist_ok=True)

    log_rows = [["epoch", "train_loss", "val_loss", "epoch_time_sec"]]

    start_all = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for i, (pixels, labels) in enumerate(train_loader):
            pixels, labels = pixels.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                outputs = model(pixel_values=pixels, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for pixels, labels in val_loader:
                pixels, labels = pixels.to(dev), labels.to(dev)
                with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                    outputs = model(pixel_values=pixels, labels=labels)
                    total_val_loss += outputs.loss.item()
        avg_val = total_val_loss / len(val_loader)
        epoch_dur = round(time.time() - epoch_start, 1)

        print(f"--> Epoch {epoch} Complete | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Duration: {epoch_dur}s")
        log_rows.append([epoch, round(avg_train, 4), round(avg_val, 4), epoch_dur])

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Save to main checkpoints
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            processor.save_pretrained(args.save_dir)
            # Sync to webapp checkpoints
            model.save_pretrained(args.webapp_save_dir)
            tokenizer.save_pretrained(args.webapp_save_dir)
            processor.save_pretrained(args.webapp_save_dir)
            print(f"    [Checkpoint] Saved best checkpoint (Val Loss: {avg_val:.4f}) to {args.save_dir}")

    with open(args.log_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(log_rows)

    total_time = round(time.time() - start_all, 1)
    print(f"\n[Finished] ViT-GPT2 Fine-Tuning Completed in {total_time}s! Checkpoints ready in webapp.")

if __name__ == "__main__":
    main()
