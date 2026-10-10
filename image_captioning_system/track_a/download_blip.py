import sys
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID = "Salesforce/blip-image-captioning-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Target device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
print(f"Downloading/loading processor for {MODEL_ID}...", flush=True)
processor = BlipProcessor.from_pretrained(MODEL_ID)
print("Processor successfully loaded!", flush=True)

print(f"Downloading/loading model weights for {MODEL_ID} (~990MB)...", flush=True)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()
print("Model weights successfully loaded onto GPU!", flush=True)

# Quick verification with a dummy image
img = Image.new("RGB", (224, 224), color=(100, 150, 200))
inputs = processor(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20)
caption = processor.decode(out[0], skip_special_tokens=True).strip()
print(f"Verification test passed! Output caption: '{caption}'", flush=True)
