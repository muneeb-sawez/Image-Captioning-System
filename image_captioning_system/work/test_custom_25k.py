import os
import re
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

model_name = 'nlpconnect/vit-gpt2-image-captioning'
ckpt_path = r'c:\Users\munee\OneDrive\Desktop\DS LAB MINI PROJECT\image_captioning_system\checkpoints\vit_gpt2_flickr30k_25k.pt'

print("Loading model and custom 25k weights...")
model = VisionEncoderDecoderModel.from_pretrained(model_name).to('cuda').eval()
ckpt = torch.load(ckpt_path, map_location='cuda')
model.load_state_dict(ckpt['model_state_dict'])

processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

test_dir = r'c:\Users\munee\OneDrive\Desktop\DS LAB MINI PROJECT\image_captioning_system\track_a\test_images'
test_imgs = [
    '1_person_reading.jpg',
    '2_animal_dog.jpg',
    '3_scene_forest_mountains.jpg',
    '4_object_vintage_car.jpg',
    '5_complex_cityscape.jpg'
]

print("\n--- RESULTS FROM CUSTOM 25K FLICKR30K MODEL ---")
for img_name in test_imgs:
    p = os.path.join(test_dir, img_name)
    if not os.path.exists(p):
        continue
    img = Image.open(p).convert('RGB')
    pv = processor(images=[img], return_tensors='pt').pixel_values.to('cuda')
    out = model.generate(
        pv,
        max_length=18,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    clean = re.sub(r'["\']', '', raw)
    sents = [x.strip() for x in re.split(r'[.!?]+', clean) if x.strip()]
    cap = (sents[0][0].upper() + sents[0][1:] + '.') if sents else 'None'
    print(f"[{img_name}]: {cap}")
