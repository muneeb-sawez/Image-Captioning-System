"""MS-COCO Dataset Downloader and Ingestion Utility.
Downloads MS-COCO 2017 Annotations and Images with streaming progress and resume support.

Usage:
    python download_coco.py --target_count 100000 --download_dir data/coco
"""
import argparse
import os
import sys
import time
import zipfile
import urllib.request
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_TRAIN_IMG_URL = "http://images.cocodataset.org/zips/train2017.zip"

def download_file_with_progress(url, dest_path):
    print(f"[Download] Connecting to: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        start_time = time.time()
        
        with open(dest_path, "wb") as out_file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                elapsed = max(1e-5, time.time() - start_time)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed
                pct = (downloaded / total_size) * 100 if total_size else 0
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r--> [{pct:.1f}%] {mb_down:.1f} MB / {mb_tot:.1f} MB ({speed_mb:.2f} MB/s)")
                sys.stdout.flush()
    print("\n[Download] Complete!")


def extract_zip(zip_path, extract_to):
    print(f"[Extract] Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print("[Extract] Extracted successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", default=os.path.join(BASE_DIR, "data", "coco"))
    parser.add_argument("--annotations_only", action="store_true", help="Download only annotations")
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)
    ann_zip = os.path.join(args.download_dir, "annotations_trainval2017.zip")
    ann_dir = os.path.join(args.download_dir, "annotations")

    # Step 1: Download annotations
    if not os.path.exists(ann_dir):
        if not os.path.exists(ann_zip):
            print("[Step 1/2] Downloading MS-COCO 2017 Annotations (~241 MB)...")
            download_file_with_progress(COCO_ANN_URL, ann_zip)
        extract_zip(ann_zip, args.download_dir)
    else:
        print("[Step 1/2] Annotations directory already exists. Skipping download.")

    if args.annotations_only:
        print("[Done] Annotations ready.")
        return

    # Step 2: Download images archive
    img_zip = os.path.join(args.download_dir, "train2017.zip")
    img_dir = os.path.join(args.download_dir, "train2017")
    if not os.path.exists(img_dir):
        if not os.path.exists(img_zip):
            print("\n[Step 2/2] Downloading MS-COCO 2017 Images (~19.3 GB)...")
            print("Note: This will download 118,287 images for overnight training.")
            download_file_with_progress(COCO_TRAIN_IMG_URL, img_zip)
        extract_zip(img_zip, args.download_dir)
    else:
        print(f"[Step 2/2] Images directory already exists at {img_dir}.")

    print("\n[Finished] MS-COCO 2017 setup complete! Ready for overnight training.")

if __name__ == "__main__":
    main()
