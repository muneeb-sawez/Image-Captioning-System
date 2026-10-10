# Caption Lens

Run:
    pip install -r requirements.txt
    python app.py        # then open http://127.0.0.1:5000

The first BLIP caption downloads the model (about 1 GB). The GRU column stays empty until checkpoints/gru_best.pt exists.

Data files the pages read (all optional, pages show a hint when missing):
- results/metrics.json: copy metrics.example.json and fill in numbers (BLEU on a 0-1 scale)
- results/train_log.csv: columns epoch,train_loss,val_loss
- results/comparison.csv: columns image,human,blip,gru
- data/glossary.json: list of {"term", "definition"}
- model.py: your CaptionGRU class (needed by gru_caption.py)
