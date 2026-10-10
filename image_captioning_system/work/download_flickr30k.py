"""
Bulletproof Resumable Multi-Threaded Downloader for Flickr30k.
- 6 parallel worker streams with automatic range-resume on disconnect.
- Auto-retries on socket drop.
- Validates exact byte count before assembly and extraction.
"""

import os
import sys
import time
import zipfile
import urllib.request
import concurrent.futures

DATASET_URL = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"
TARGET_DIR = r"c:\Users\munee\OneDrive\Desktop\DS LAB MINI PROJECT\image_captioning_system\data\flickr30k"
ZIP_PATH = os.path.join(TARGET_DIR, "flickr30k-images.zip")
IMAGES_DIR = os.path.join(TARGET_DIR, "Images")
NUM_WORKERS = 6
TOTAL_EXPECTED_SIZE = 4390240817  # Exact bytes from Hugging Face CDN


def download_chunk_resumable(url, start_byte, end_byte, part_path, chunk_id):
    expected_len = end_byte - start_byte + 1
    max_retries = 25
    retries = 0

    while retries < max_retries:
        downloaded = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if downloaded >= expected_len:
            return True

        current_start = start_byte + downloaded
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes={current_start}-{end_byte}")
        req.add_header("User-Agent", "Mozilla/5.0")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(part_path, "ab") as f:
                    while True:
                        buf = resp.read(1024 * 512)  # 512KB
                        if not buf:
                            break
                        f.write(buf)
            downloaded = os.path.getsize(part_path)
            if downloaded >= expected_len:
                return True
        except Exception as e:
            retries += 1
            time.sleep(2)

    raise RuntimeError(f"Part {chunk_id} failed after {max_retries} retries.")


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    total_gb = TOTAL_EXPECTED_SIZE / (1024 ** 3)
    
    chunk_size = TOTAL_EXPECTED_SIZE // NUM_WORKERS
    tasks = []
    part_files = []
    
    for i in range(NUM_WORKERS):
        start = i * chunk_size
        end = TOTAL_EXPECTED_SIZE - 1 if i == NUM_WORKERS - 1 else (i + 1) * chunk_size - 1
        part_path = os.path.join(TARGET_DIR, f"part_{i}.tmp")
        part_files.append((part_path, end - start + 1))
        tasks.append((DATASET_URL, start, end, part_path, i))

    print(f"[*] Resumable download active across {NUM_WORKERS} streams (Total: {total_gb:.2f} GB)...", flush=True)
    t0 = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_chunk = {executor.submit(download_chunk_resumable, *t): t[4] for t in tasks}
        
        while not all(f.done() for f in future_to_chunk):
            time.sleep(8)
            downloaded = sum(os.path.getsize(p) for p, _ in part_files if os.path.exists(p))
            elapsed = time.time() - t0
            speed = (downloaded / (1024 * 1024)) / max(elapsed, 1)
            pct = (downloaded / TOTAL_EXPECTED_SIZE) * 100
            rem_gb = (TOTAL_EXPECTED_SIZE - downloaded) / (1024 ** 3)
            eta_min = (rem_gb * 1024) / max(speed, 0.1) / 60
            print(f"[*] Progress: {pct:.1f}% ({downloaded / (1024**3):.2f}/{total_gb:.2f} GB) | Speed: {speed:.2f} MB/s | ETA: {eta_min:.1f} mins", flush=True)

        # Check all results
        for f in future_to_chunk:
            f.result()

    print("[*] All parts verified 100% complete! Assembling full zip archive...", flush=True)
    with open(ZIP_PATH, "wb") as out_f:
        for p, expected_len in part_files:
            if os.path.getsize(p) != expected_len:
                raise ValueError(f"Chunk size mismatch in {p}!")
            with open(p, "rb") as in_f:
                while True:
                    buf = in_f.read(32 * 1024 * 1024)
                    if not buf:
                        break
                    out_f.write(buf)
            os.remove(p)
            
    final_size = os.path.getsize(ZIP_PATH)
    print(f"[*] Zip assembled cleanly: {ZIP_PATH} ({final_size / (1024**3):.2f} GB)", flush=True)
    if final_size != TOTAL_EXPECTED_SIZE:
        raise ValueError(f"Size mismatch: {final_size} vs {TOTAL_EXPECTED_SIZE}")
        
    print("[*] Extracting images into data/flickr30k/Images...", flush=True)
    t_extract = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(IMAGES_DIR)
        
    print(f"[*] Extraction complete in {time.time() - t_extract:.1f}s!", flush=True)
    
    img_count = len([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[SUCCESS] Total images in flickr30k/Images: {img_count}", flush=True)
    
    # Delete zip to save 4.1 GB disk space
    try:
        os.remove(ZIP_PATH)
    except Exception:
        pass


if __name__ == "__main__":
    main()
