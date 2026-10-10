# Part B: CNN + 2-layer GRU captioner (trained by us)

## Data layout
Download Flickr8k (for example from Kaggle) so you have either:
- `data/flickr8k/Images/` and `data/flickr8k/captions.txt`  (CSV with header `image,caption`), or
- the original `Flicker8k_Dataset/` folder and `Flickr8k.token.txt`.

## Run (in order)
    pip install -r requirements.txt
    python prepare_data.py --captions data/flickr8k/captions.txt --images data/flickr8k/Images
    python extract_features.py      # slow step, saves work/features.pt (use Google Colab if your laptop is slow)
    python train.py                 # saves checkpoints/gru_best.pt and results/train_log.csv
    python evaluate.py --blip --limit 200   # BLEU on the test split + comparison with BLIP

Ablation: `python train.py --layers 1 --ckpt checkpoints/gru_1layer.pt --log results/train_log_1layer.csv`,
then `python evaluate.py --ckpt checkpoints/gru_1layer.pt --results results_1layer`.

## What the files do
| File | Purpose |
|---|---|
| prepare_data.py | reads captions, splits by image (80/10/10, seeded), builds vocabulary from the training split only |
| extract_features.py | frozen ResNet-50 -> 2048 numbers per image |
| model.py | `CaptionGRU`: 2-layer GRU decoder with greedy and beam search |
| train.py | teacher-forced training, early stopping, best checkpoint |
| evaluate.py | BLEU-1..4, agreement with BLIP, writes files for the website |

## Using it with the Flask website
Copy `model.py` next to the website's `app.py`, and copy `checkpoints/gru_best.pt`, `results/metrics.json`,
`results/train_log.csv` and `results/comparison.csv` into the website folder. The checkpoint format already matches.

"Agreement" = BLEU-1 of our captions against BLIP's captions (BLIP is a stand-in reference, not ground truth).
