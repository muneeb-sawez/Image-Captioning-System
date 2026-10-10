"""Inference for our CNN + GRU model.
Expects checkpoints/gru_best.pt saved by train.py as:
  {"model_state": model.state_dict(), "itos": [word, ...],
   "config": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2}}
and a model.py that defines CaptionGRU (see the project plan, Appendix A.3)."""
import os, torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from model import CaptionGRU

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "gru_best.pt")
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load():
    if not os.path.exists(CKPT):
        return None
    ck = torch.load(CKPT, map_location=dev)
    itos = ck["itos"]; stoi = {w: i for i, w in enumerate(itos)}
    gru = CaptionGRU(len(itos), **ck.get("config", {})).to(dev)
    gru.load_state_dict(ck["model_state"]); gru.eval()
    cnn = resnet50(weights=ResNet50_Weights.DEFAULT); cnn.fc = torch.nn.Identity(); cnn = cnn.to(dev).eval()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    start, end = stoi["<start>"], stoi["<end>"]

    @torch.no_grad()
    def caption(img):
        feats = cnn(tf(img).unsqueeze(0).to(dev))
        words = []
        for i in gru.generate(feats, start, end)[0].tolist():
            if i == end: break
            words.append(itos[i])
        return " ".join(words)
    return caption
