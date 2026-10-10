import csv
import json
import os
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB upload limit

HYPERPARAMS = [
    {"param": "Feature Extractor", "value": "ResNet-50 (Pretrained)", "detail": "Frozen weights, 2048-d bottleneck embedding"},
    {"param": "Embedding Dimension", "value": "256", "detail": "Learned word representation space"},
    {"param": "Hidden Dimension", "value": "512", "detail": "State size per GRU cell layer"},
    {"param": "Recurrent Layers", "value": "2", "detail": "Stacked Gated Recurrent Units"},
    {"param": "Dropout Rate", "value": "0.5", "detail": "Applied between layers and to outputs"},
    {"param": "Optimization", "value": "Adam (lr = 3e-4)", "detail": "Teacher forcing with cross-entropy loss"},
    {"param": "Gradient Clipping", "value": "5.0", "detail": "Prevents exploding gradients in RNN"},
    {"param": "Batch Size & Patience", "value": "64 / 3 Epochs", "detail": "Early stopping on validation loss"},
    {"param": "Dataset & Splits", "value": "Flickr8k (80/10/10)", "detail": "6,473 Train / 809 Val / 809 Test (strictly split by image)"},
]

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

@app.route("/")
def index():
    metrics = read_json("results/metrics.json", {})
    comparisons = read_csv("results/comparison.csv")
    preview_comparisons = comparisons[:8] if comparisons else []
    train_log = read_csv("results/train_log.csv")
    return render_template(
        "index.html",
        metrics=metrics,
        samples=preview_comparisons,
        train_log=train_log,
        hyperparams=HYPERPARAMS
    )

@app.route("/research")
def research():
    metrics = read_json("results/metrics.json", {})
    train_log = read_csv("results/train_log.csv")
    try:
        max_val = max(float(r["val_loss"]) for r in train_log)
        min_val = min(float(r["val_loss"]) for r in train_log)
        for r in train_log:
            # normalized height percentage for CSS visual chart
            v = float(r["val_loss"])
            r["val_pct"] = round(15 + (v - min_val) / (max_val - min_val + 1e-6) * 75)
            t = float(r["train_loss"])
            r["train_pct"] = round(15 + (t - min_val) / (max_val - min_val + 1e-6) * 75)
    except (KeyError, ValueError):
        pass

    comparisons = read_csv("results/comparison.csv")
    return render_template(
        "research.html",
        metrics=metrics,
        train_log=train_log,
        comparisons=comparisons[:30] if comparisons else [],
        total_samples=len(comparisons),
        hyperparams=HYPERPARAMS
    )

@app.route("/about")
def about():
    return render_template("about.html")

# Dedicated Inference Studio Page
@app.route("/studio")
@app.route("/demo")
def studio():
    return render_template("studio.html")

@app.route("/how")
def redirect_how(): return redirect("/#how")

@app.route("/models")
def redirect_models(): return redirect("/research#models")

@app.route("/results")
def redirect_results(): return redirect("/research#evaluation")

@app.route("/glossary")
def redirect_glossary(): return redirect("/research#architecture")

# ---- Model singletons loaded on first inference ----
_cache = {}

def get_blip():
    if "blip" not in _cache:
        from blip_caption import generate_caption
        _cache["blip"] = generate_caption
    return _cache["blip"]

def get_gru():
    if "gru" not in _cache:
        import gru_caption
        _cache["gru"] = gru_caption.load()
    return _cache["gru"]

def get_vit():
    if "vit" not in _cache:
        import vit_gpt2_caption
        _cache["vit"] = vit_gpt2_caption.load()
    return _cache["vit"]

@app.post("/api/caption")
def caption():
    start_time = time.time()
    img = None
    
    # 1. Check for uploaded file
    if "image" in request.files and request.files["image"].filename:
        try:
            img = Image.open(request.files["image"].stream).convert("RGB")
        except Exception:
            return jsonify(error="Invalid image format uploaded."), 400
            
    # 2. Check for sample image name
    elif request.form.get("sample"):
        sample_name = request.form.get("sample").strip().replace("..", "").replace("/", "")
        sample_path = os.path.join(ROOT, "static", "samples", sample_name)
        if os.path.exists(sample_path):
            try:
                img = Image.open(sample_path).convert("RGB")
            except Exception:
                return jsonify(error="Could not open sample image."), 400
        else:
            return jsonify(error=f"Sample '{sample_name}' not found."), 404

    if img is None:
        return jsonify(error="No image file or sample provided."), 400

    beams = max(1, min(10, request.form.get("beams", 5, type=int)))
    prompt = request.form.get("prompt", "").strip() or None
    out = {"blip": None, "vit": None, "gru": None, "timing": {}}

    # Inference: BLIP
    t0 = time.time()
    try:
        blip_fn = get_blip()
        out["blip"] = blip_fn(img, num_beams=beams, prompt=prompt)
        out["timing"]["blip_ms"] = round((time.time() - t0) * 1000)
    except Exception as e:
        app.logger.exception("BLIP inference error")
        out["blip"] = f"BLIP generation error: {str(e)}"

    # Inference: Vision Transformer + GPT-2
    t1 = time.time()
    try:
        vit_fn = get_vit()
        out["vit"] = vit_fn(img, num_beams=beams)
        out["timing"]["vit_ms"] = round((time.time() - t1) * 1000)
    except Exception as e:
        app.logger.exception("ViT-GPT2 inference error")
        out["vit"] = f"ViT-GPT2 error: {str(e)}"

    # Inference: Custom GRU Baseline
    t2 = time.time()
    try:
        gru_fn = get_gru()
        if gru_fn:
            out["gru"] = gru_fn(img, beam=beams)
            out["timing"]["gru_ms"] = round((time.time() - t2) * 1000)
        else:
            out["gru"] = "Checkpoint gru_best.pt not found on server."
    except Exception as e:
        app.logger.exception("GRU inference error")
        out["gru"] = f"GRU generation error: {str(e)}"

    out["timing"]["total_ms"] = round((time.time() - start_time) * 1000)
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
