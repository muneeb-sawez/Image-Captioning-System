"""Step 2: run every image through a frozen, pre-trained ResNet-50 once and save the 2048-number features.

This is the slow step. On a laptop CPU it can take a while; free Google Colab is much faster.
The first run downloads the ResNet-50 weights.

Usage:
    python extract_features.py
"""
import argparse
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

from common import read_json


class Images(Dataset):
    def __init__(self, names, root):
        self.names, self.root = names, root
        self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.tf(Image.open(os.path.join(self.root, self.names[i])).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    caps = read_json(os.path.join(a.work, "captions.json"))
    root = read_json(os.path.join(a.work, "meta.json"))["images"]
    names = sorted({n for split in caps.values() for n in split})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
    cnn.fc = torch.nn.Identity()  # drop the classifier, keep the 2048-d features
    cnn = cnn.to(device).eval()

    loader = DataLoader(Images(names, root), batch_size=a.batch, num_workers=a.workers)
    feats = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats.append(cnn(batch.to(device)).cpu())
            if i % 10 == 0:
                print(f"{min((i + 1) * a.batch, len(names))}/{len(names)} images", flush=True)

    torch.save({"names": names, "feats": torch.cat(feats)}, os.path.join(a.work, "features.pt"))
    print(f"Saved features for {len(names)} images on {device}.")


if __name__ == "__main__":
    main()
