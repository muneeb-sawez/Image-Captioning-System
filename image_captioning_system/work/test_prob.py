import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import os, sys

sys.path.insert(0, 'image_captioning_system/webapp')
from model import CaptionGRU

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load('image_captioning_system/webapp/checkpoints/gru_best.pt', map_location=dev)
itos = ck['itos']
stoi = {w: i for i, w in enumerate(itos)}
gru = CaptionGRU(len(itos), **ck.get('config', {})).to(dev)
gru.load_state_dict(ck['model_state'])
gru.eval()

tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn.fc = torch.nn.Identity()
cnn = cnn.to(dev).eval()

img = Image.open('image_captioning_system/webapp/static/samples/beach.jpg').convert('RGB')
feats = cnn(tf(img).unsqueeze(0).to(dev))

h = gru.init_hidden(feats)
v_proj = torch.relu(gru.feat_proj(feats)).unsqueeze(1)
tok = torch.tensor([[stoi['<start>']]], dtype=torch.long, device=dev)

print('--- Step-by-step greedy on beach.jpg ---')
for step in range(12):
    w_emb = gru.embed(tok)
    gru_input = torch.cat([w_emb, v_proj], dim=-1)
    out, h = gru.gru(gru_input, h)
    logits = gru.fc(out[:, -1])
    probs = torch.softmax(logits, dim=-1)[0]
    top5 = torch.topk(probs, 5)
    cand_str = ', '.join([f'{itos[idx]}:{val.item()*100:.1f}%' for val, idx in zip(top5.values, top5.indices)])
    tok = logits.argmax(-1, keepdim=True)
    w = itos[tok.item()]
    print(f'Step {step}: chosen="{w}" | Top candidates: {cand_str}')
    if w == '<end>': break
