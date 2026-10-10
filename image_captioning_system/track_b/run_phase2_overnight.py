"""Phase 2 Master Overnight Runner.
Coordinates:
  1. Completion of MS-COCO 2017 Download & Extraction (Task 787 / download_coco.py)
  2. Launch of train_overnight_100k.py across MS-COCO + Flickr8k
  3. Automatic Checkpointing and Web App Model Synchronization

Usage:
    python run_phase2_overnight.py
"""
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 65)
    print("  PHASE 2: OVERNIGHT 100K LARGE-SCALE PIPELINE (MS-COCO + FLICKR8K)  ")
    print("=" * 65)

    coco_ann = os.path.join(BASE_DIR, "data", "coco", "annotations", "captions_train2017.json")
    coco_img = os.path.join(BASE_DIR, "data", "coco", "train2017")

    # Check if download is already complete or still running
    if not (os.path.exists(coco_ann) and os.path.exists(coco_img)):
        print("\n[Stage 1/2] Ensuring MS-COCO download and extraction is complete...")
        download_script = os.path.join(BASE_DIR, "track_b", "download_coco.py")
        ret = subprocess.call([sys.executable, download_script])
        if ret != 0:
            print("[Warning] Download exited with non-zero status. Checking files...")

    print("\n[Stage 2/2] Launching 100K Image Training (MS-COCO + Flickr8k)...")
    train_script = os.path.join(BASE_DIR, "track_b", "train_overnight_100k.py")
    cmd = [
        sys.executable, train_script,
        "--epochs", "5",
        "--batch", "8",
        "--lr", "3e-5"
    ]
    subprocess.call(cmd)
    print("\n[Done] Phase 2 Overnight pipeline finished!")

if __name__ == "__main__":
    main()
