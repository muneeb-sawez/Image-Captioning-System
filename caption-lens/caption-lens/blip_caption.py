import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID = "Salesforce/blip-image-captioning-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()

def generate_caption(image: Image.Image, num_beams: int = 5, max_new_tokens: int = 40) -> str:
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True).strip()
