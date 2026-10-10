"""Inference module for Vision Transformer + GPT-2 Cross-Attention Model.
Provides open-domain captioning with patch-level visual grounding.
"""
import os
import re
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
_dev = "cuda" if torch.cuda.is_available() else "cpu"

def load():
    custom_ckpt = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "vit_gpt2_flickr30k_25k.pt")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(_dev).eval()
    
    if os.path.exists(custom_ckpt):
        print(f"[ViT-GPT2] Loading custom fine-tuned weights from: {custom_ckpt} on {_dev}")
        ckpt_data = torch.load(custom_ckpt, map_location=_dev)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        model.load_state_dict(state_dict)
    else:
        print(f"[ViT-GPT2] Loading pristine pretrained weights from: {MODEL_NAME} on {_dev}")
        
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    @torch.no_grad()
    def caption(img: Image.Image, num_beams: int = 4, max_length: int = 28) -> str:
        if img.mode != "RGB":
            img = img.convert("RGB")
        pixel_values = processor(images=[img], return_tensors="pt").pixel_values.to(_dev)
        output_ids = model.generate(
            pixel_values,
            max_length=max_length,
            min_length=8,
            num_beams=max(1, min(num_beams, 8)),
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if not raw:
            return "No description generated."
        
        # Clean quotes
        clean = re.sub(r'["\']', '', raw).strip()
        
        # If there is a terminal period, prefer the first complete sentence
        if '.' in clean:
            first_sent = clean.split('.')[0].strip()
        else:
            first_sent = clean
            
        # Strip any dangling connectors/prepositions if cut off mid-thought
        dangling_pattern = r'\b(and|or|with|in|at|on|to|for|of|the|a|an|is|are|has|while|that|as|by)\s*$'
        while re.search(dangling_pattern, first_sent, flags=re.IGNORECASE):
            first_sent = re.sub(dangling_pattern, '', first_sent, flags=re.IGNORECASE).strip()
            
        if not first_sent:
            return "No description generated."
            
        # Capitalize and format with clean terminal period
        final_caption = first_sent[0].upper() + first_sent[1:] + "."
        return final_caption

    return caption
