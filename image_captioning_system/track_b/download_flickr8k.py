"""Download and extract Flickr8k dataset from Hugging Face directly into data/flickr8k/"""
import os
import zipfile
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "shivangibithel/Flickr8k"
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "flickr8k"))
os.makedirs(TARGET_DIR, exist_ok=True)

print(f"Target directory: {TARGET_DIR}")

# 1. Download captions.txt
print("Downloading captions.txt (~3.2 MB)...", flush=True)
captions_path = hf_hub_download(repo_id=REPO_ID, filename="captions.txt", repo_type="dataset")
dest_captions = os.path.join(TARGET_DIR, "captions.txt")
shutil.copyfile(captions_path, dest_captions)
print(f"Saved captions to: {dest_captions}", flush=True)

# 2. Download Images.zip
print("Downloading Images.zip (~1.05 GB)... This may take a few minutes depending on internet speed.", flush=True)
zip_path = hf_hub_download(repo_id=REPO_ID, filename="Images.zip", repo_type="dataset")
print(f"Downloaded zip to cache: {zip_path}", flush=True)

# 3. Extract Images
images_dest = os.path.join(TARGET_DIR, "Images")
os.makedirs(images_dest, exist_ok=True)
print(f"Extracting Images.zip into {images_dest}...", flush=True)

with zipfile.ZipFile(zip_path, "r") as zf:
    members = zf.namelist()
    print(f"Total entries in archive: {len(members)}", flush=True)
    # Check if files inside zip already have an 'Images/' prefix or not
    prefix = ""
    for m in members:
        if m.endswith((".jpg", ".png", ".jpeg")):
            # extract directly into images_dest
            base_name = os.path.basename(m)
            if base_name:
                target_file = os.path.join(images_dest, base_name)
                if not os.path.exists(target_file):
                    with zf.open(m) as src, open(target_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)

num_extracted = len([f for f in os.listdir(images_dest) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
print(f"\nExtraction complete! Found {num_extracted} images in {images_dest}")
print(f"Captions file size: {os.path.getsize(dest_captions):,} bytes")
print("Flickr8k setup is ready!")
