import os
import json
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T

work_dir = os.path.join(os.path.dirname(__file__), "..", "work")
img_dir = os.path.join(os.path.dirname(__file__), "..", "data", "flickr8k", "Images")

# 1. Update Vocab
vocab_path = os.path.join(work_dir, "vocab.json")
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

new_words = ["signature", "blazer", "conference", "cursive", "illuminated", "formal", "lobby", "stage", "navy"]
added = 0
for w in new_words:
    if w not in vocab["itos"]:
        vocab["itos"].append(w)
        added += 1

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f)
print(f"Added {added} new words to vocab. Total vocab: {len(vocab['itos'])}")

# 2. Extract ResNet-50 Features for the 4 new images
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading ResNet-50 on {device}...")
cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn.fc = torch.nn.Identity()
cnn = cnn.to(device).eval()

tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

new_images = ["muneeb_suit.jpg", "muneeb_office.jpg", "muneeb_conference.jpg", "muneeb_signature.jpg"]
new_feats = []
for img_name in new_images:
    p = os.path.join(img_dir, img_name)
    im = Image.open(p).convert("RGB")
    t = tf(im).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = cnn(t).cpu()
    new_feats.append(feat)

new_feats = torch.cat(new_feats, dim=0)

# 3. Update features.pt
feat_path = os.path.join(work_dir, "features.pt")
fd = torch.load(feat_path, map_location="cpu")

for idx, img_name in enumerate(new_images):
    if img_name not in fd["names"]:
        fd["names"].append(img_name)
        fd["feats"] = torch.cat([fd["feats"], new_feats[idx:idx+1]], dim=0)

torch.save(fd, feat_path)
print(f"features.pt updated: now {len(fd['names'])} images, tensor shape {fd['feats'].shape}")

# 4. Update captions.json
caps_path = os.path.join(work_dir, "captions.json")
with open(caps_path, "r", encoding="utf-8") as f:
    caps = json.load(f)

custom_caps = {
    "muneeb_suit.jpg": [
        ["a", "young", "man", "wearing", "glasses", "and", "a", "dark", "blue", "suit", "with", "a", "tie"],
        ["a", "smiling", "man", "in", "a", "business", "suit", "and", "blue", "tie"],
        ["a", "man", "with", "glasses", "smiling", "in", "a", "formal", "dark", "suit"],
        ["a", "portrait", "of", "a", "young", "man", "wearing", "glasses", "and", "a", "suit"],
        ["a", "professional", "portrait", "of", "a", "man", "in", "a", "navy", "suit", "and", "tie"]
    ],
    "muneeb_office.jpg": [
        ["a", "smiling", "man", "wearing", "round", "glasses", "and", "a", "dark", "blazer"],
        ["a", "young", "man", "with", "glasses", "and", "a", "blazer", "in", "an", "office"],
        ["a", "man", "wearing", "a", "dark", "jacket", "and", "white", "shirt", "smiling", "indoors"],
        ["a", "portrait", "of", "a", "young", "man", "with", "glasses", "in", "an", "office", "setting"],
        ["a", "man", "in", "glasses", "and", "a", "dark", "suit", "smiling", "at", "the", "camera"]
    ],
    "muneeb_conference.jpg": [
        ["a", "man", "standing", "in", "front", "of", "a", "conference", "sign", "with", "arms", "open"],
        ["a", "young", "man", "standing", "in", "a", "hotel", "lobby", "at", "an", "annual", "conference"],
        ["a", "man", "in", "a", "polo", "shirt", "and", "pants", "at", "a", "conference", "event"],
        ["a", "man", "standing", "before", "an", "illuminated", "conference", "entrance"],
        ["a", "person", "posing", "in", "front", "of", "a", "conference", "sign", "in", "an", "event", "hall"]
    ],
    "muneeb_signature.jpg": [
        ["a", "handwritten", "signature", "in", "black", "ink", "on", "white", "paper"],
        ["a", "person", "name", "written", "in", "cursive", "black", "ink"],
        ["a", "cursive", "handwritten", "signature", "on", "a", "blank", "white", "page"],
        ["black", "ink", "writing", "of", "a", "signature", "on", "white", "paper"],
        ["a", "handwritten", "name", "written", "with", "a", "black", "pen", "on", "paper"]
    ]
}

for img_name, c_list in custom_caps.items():
    caps["train"][img_name] = c_list

with open(caps_path, "w", encoding="utf-8") as f:
    json.dump(caps, f)

print(f"captions.json updated: {len(caps['train'])} training images.")
