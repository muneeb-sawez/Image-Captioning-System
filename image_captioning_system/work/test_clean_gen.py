import re
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
base_model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning').to(dev).eval()
processor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

for sample in ['car.jpg', 'city.jpg', 'beach.jpg', 'reading.jpg', 'dog.jpg']:
    img = Image.open(f'image_captioning_system/webapp/static/samples/{sample}').convert('RGB')
    pv = processor(images=[img], return_tensors='pt').pixel_values.to(dev)
    ids = base_model.generate(pv, max_length=18, num_beams=4, no_repeat_ngram_size=2, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    clean = re.sub(r'["\']', '', raw)
    s = [x.strip() for x in re.split(r'[.!?]+', clean) if x.strip()]
    res = (s[0][0].upper() + s[0][1:] + '.') if s else 'No description generated.'
    print(f'{sample:12} -> {res}')
