import csv, json, os
from flask import Flask, render_template, request, jsonify
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

NAV = [("index", "Home"), ("demo", "Demo"), ("how", "How it works"), ("models", "Models"),
       ("results", "Results"), ("glossary", "Glossary"), ("about", "About")]
HYPERPARAMS = {"Embedding size": 256, "Hidden size": 512, "GRU layers": 2, "Dropout": 0.5,
               "Optimizer": "Adam, learning rate 3e-4", "Gradient clipping": 5, "Batch size": 64, "Epochs": "about 20"}

def read_json(path, default):
    try:
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default

def read_csv(path):
    try:
        with open(os.path.join(ROOT, path), encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []

@app.context_processor
def inject_nav():
    return {"nav": NAV}

@app.route("/")
def index(): return render_template("index.html")

@app.route("/demo")
def demo(): return render_template("demo.html")

@app.route("/how")
def how(): return render_template("how.html")

@app.route("/models")
def models():
    log = read_csv("results/train_log.csv")
    try:
        top = max(float(r["val_loss"]) for r in log)
        for r in log: r["pct"] = round(float(r["val_loss"]) / top * 100)
    except (KeyError, ValueError):
        log = []
    return render_template("models.html", hp=HYPERPARAMS, log=log)

@app.route("/results")
def results():
    return render_template("results.html", metrics=read_json("results/metrics.json", {}),
                           rows=read_csv("results/comparison.csv")[:15])

@app.route("/glossary")
def glossary(): return render_template("glossary.html", terms=read_json("data/glossary.json", []))

@app.route("/about")
def about(): return render_template("about.html")

# ---- models are loaded on first use, so the pages work without torch installed ----
_cache = {}
def get_blip():
    if "blip" not in _cache:
        from blip_caption import generate_caption
        _cache["blip"] = generate_caption
    return _cache["blip"]

def get_gru():
    if "gru" not in _cache:
        import gru_caption
        _cache["gru"] = gru_caption.load()  # None if no checkpoint yet
    return _cache["gru"]

@app.post("/api/caption")
def caption():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No image uploaded."), 400
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception:
        return jsonify(error="That file is not a valid image."), 400
    beams = max(1, min(10, request.form.get("beams", 5, type=int)))
    out = {"blip": None, "gru": None}
    try:
        out["blip"] = get_blip()(img, num_beams=beams)
    except Exception:
        app.logger.exception("BLIP failed")
    try:
        g = get_gru()
        out["gru"] = g(img) if g else None
    except Exception:
        app.logger.exception("GRU failed")
    if not (out["blip"] or out["gru"]):
        return jsonify(error="No model is available. Check the server console."), 500
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
