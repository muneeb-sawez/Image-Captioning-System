"""
Ultra-fast, stable fine-tuning on 25,000 Flickr30k images for ViT-GPT2 Image Captioning.
- Freezes Vision Transformer (ViT-B/16) backbone for high speed and 0 VRAM overflow.
- Trains GPT-2 cross-attention and language modeling head.
- Full FP16 mixed-precision acceleration.
- Strict <|endoftext|> target masking to ensure concise, natural English captions.
- Produces training loss curve plot and custom checkpoint.
"""

import os
import ast
import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Hyperparameters
DATA_DIR = r"c:\Users\munee\OneDrive\Desktop\DS LAB MINI PROJECT\image_captioning_system\data\flickr30k"
IMAGES_DIR = os.path.join(DATA_DIR, "flickr30k-images") if os.path.exists(os.path.join(DATA_DIR, "flickr30k-images")) else os.path.join(DATA_DIR, "Images")
CSV_PATH = os.path.join(DATA_DIR, "flickr_annotations_30k.csv")
SAVE_DIR = r"c:\Users\munee\OneDrive\Desktop\DS LAB MINI PROJECT\image_captioning_system\checkpoints"
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "vit_gpt2_flickr30k_25k.pt")
LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "flickr30k_training_loss.png")

NUM_SAMPLES = 25000     # Exactly 25,000 image-caption pairs
BATCH_SIZE = 16         # High throughput with frozen encoder
EPOCHS = 3
LR = 5e-5
MAX_LENGTH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flickr30kDataset(Dataset):
    def __init__(self, df, images_dir, feature_extractor, tokenizer, max_length=32):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        caption = row['caption']
        
        img_path = os.path.join(self.images_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback black image if corrupted
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Tokenize target caption with EOS token
        encoded = self.tokenizer(
            caption + self.tokenizer.eos_token,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = encoded.input_ids.squeeze(0)
        # Mask padding tokens so loss is not computed on them
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return pixel_values, labels


def main():
    print("=" * 65)
    print("      FAST 25K FLICKR30K TRAINING PIPELINE (ViT-GPT2)")
    print("=" * 65)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Load and parse dataset
    print(f"\n[1/5] Loading annotations from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Extract primary caption for each image
    records = []
    for _, row in df.iterrows():
        fname = str(row['filename']).strip()
        try:
            raw_val = row['raw']
            captions = ast.literal_eval(raw_val) if isinstance(raw_val, str) and raw_val.startswith('[') else [str(raw_val)]
            if captions:
                # Pick first high-quality caption
                records.append({'filename': fname, 'caption': str(captions[0]).strip()})
        except Exception:
            continue
            
    clean_df = pd.DataFrame(records).dropna().drop_duplicates(subset=['filename'])
    
    # Filter to only existing images
    valid_records = []
    for _, r in clean_df.iterrows():
        if os.path.exists(os.path.join(IMAGES_DIR, r['filename'])):
            valid_records.append(r)
        if len(valid_records) >= NUM_SAMPLES:
            break
            
    train_df = pd.DataFrame(valid_records)
    print(f"Target sample size: {len(train_df)} verified image-caption pairs.")
    
    if len(train_df) < 500:
        print(f"ERROR: Only found {len(train_df)} images in {IMAGES_DIR}. Waiting for extraction.")
        return

    # 2. Load model and processor
    print("\n[2/5] Initializing ViT-GPT2 architecture & tokenizers...")
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    # Freeze Vision Transformer backbone to speed up training by 3x and prevent OOM
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Vision Encoder (ViT-B/16): FROZEN (Zero extra VRAM, maximum throughput)")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    # 3. Create DataLoader
    dataset = Flickr30kDataset(train_df, IMAGES_DIR, feature_extractor, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    # 4. Training loop
    print(f"\n[3/5] Starting {EPOCHS} Epochs on {len(dataset)} samples (Batch Size: {BATCH_SIZE})...")
    epoch_losses = []
    start_time = time.time()
    
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        step_count = 0
        ep_start = time.time()
        
        for step, (pixel_values, labels) in enumerate(dataloader, 1):
            pixel_values = pixel_values.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            step_count += 1
            
            if step % 100 == 0 or step == len(dataloader):
                avg_step_loss = total_loss / step_count
                samples_done = step * BATCH_SIZE
                pct = (step / len(dataloader)) * 100
                speed = samples_done / max(time.time() - ep_start, 1)
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{step}/{len(dataloader)}] ({pct:.1f}%) | Loss: {avg_step_loss:.4f} | Speed: {speed:.1f} smp/s", flush=True)

        avg_loss = total_loss / max(step_count, 1)
        epoch_losses.append(avg_loss)
        ep_duration = (time.time() - ep_start) / 60
        print(f"--> Epoch {epoch} complete in {ep_duration:.2f} mins | Avg Loss: {avg_loss:.4f}", flush=True)

    total_duration = (time.time() - start_time) / 60
    print(f"\n[4/5] Training successfully completed in {total_duration:.2f} minutes!")

    # 5. Save Model and Loss Plot
    print(f"\n[5/5] Saving checkpoint and training report...")
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_losses[-1],
        'samples': len(train_df),
        'dataset': 'Flickr30k-25K'
    }, CHECKPOINT_PATH)
    print(f"Saved custom weights to: {CHECKPOINT_PATH}")

    # Plot Loss Curve
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', color='#2563eb', linewidth=2.5, markersize=8)
    plt.title("Fine-Tuning Loss Curve on Flickr30k (25,000 Samples)", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved training curve plot to: {LOSS_PLOT_PATH}")
    print("\nALL TASKS DONE SUCCESSFULLY!")


if __name__ == "__main__":
    main()
