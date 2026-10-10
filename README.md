# 🖼️ Image Captioning System — Deep Learning & Vision Studio

A deep-learning image captioning research and demonstration platform developed for the **Data Science Lab Mini Project**.

The system compares three different paradigm architectures for neural image captioning:
1. **ResNet-50 + 2-Layer Visually Grounded GRU** (Trained baseline with step-wise spatial grounding)
2. **Vision Transformer (ViT-B/16) + GPT-2** (Transformer cross-attention encoder-decoder)
3. **Salesforce BLIP** (Bootstrapped Language-Image Pretraining)

The project includes an interactive **Flask Web Studio** where users can upload any image or select benchmarks to generate and compare captions in real-time.

---

## 🚀 Quick Start (Run Locally in 2 Minutes)

Anyone can clone and run this web application locally without needing to download massive external datasets.

### 1. Clone the repository
```bash
git clone https://github.com/muneeb-sawez/Image-Captioning-System.git
cd Image-Captioning-System
```

### 2. Install dependencies
```bash
cd image_captioning_system/webapp
pip install -r requirements.txt
```

> **Note on PyTorch**: If you have an NVIDIA GPU with CUDA, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org), or the CPU version will automatically run seamlessly.

### 3. Launch the Web App
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- Go to the **Inference Studio** (`/studio`).
- Upload any image from your computer (or pick sample images).
- View captions generated simultaneously across all architectures!

---

## 📊 Model Evaluation & Benchmarks

All models were evaluated on the held-out benchmark test split (809 images, 2,652 vocabulary words):

### 1. BLEU Score Comparison
| Model Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | BLIP Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 + GRU (Greedy)** | **59.22%** | **41.68%** | **27.89%** | **18.61%** | 31.99% |
| **ResNet-50 + GRU (Beam=3)** | **60.41%** | **42.44%** | **29.22%** | **20.04%** | 33.44% |
| **Salesforce BLIP (Base)** | **64.53%** | **49.24%** | **37.30%** | **28.39%** | 100.0% |

### 2. Word-Level Classification Metrics (GRU)
| Metric Type | Precision | Recall | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **All Vocabulary Words (Greedy)** | 60.95% | 19.59% | 29.65% | 99.06% |
| **All Vocabulary Words (Beam)** | 63.38% | 20.12% | 30.54% | 99.08% |
| **Content Words Only (Greedy)** | 47.59% | 13.05% | 20.48% | 99.31% |
| **Content Words Only (Beam)** | 50.90% | 13.73% | 21.63% | 99.32% |

* **Token-Level Next-Word Top-1 Accuracy**: 41.20%
* **Token-Level Next-Word Top-5 Accuracy**: 69.63%
* **Perplexity**: 14.78

---

## 📁 Repository Structure

```text
├── image_captioning_system/
│   ├── webapp/                 # Interactive Flask web application
│   │   ├── app.py              # Main Flask server & API routes
│   │   ├── blip_caption.py     # BLIP inference pipeline
│   │   ├── gru_caption.py      # ResNet + GRU inference pipeline
│   │   ├── vit_gpt2_caption.py # ViT + GPT-2 inference pipeline
│   │   ├── checkpoints/        # Lightweight trained weights (gru_best.pt)
│   │   ├── templates/          # Modern UI HTML templates
│   │   └── static/             # CSS styling, JS charts, sample images
│   ├── track_a/                # Part A: Pretrained model exploration
│   ├── track_b/                # Part B: Custom CNN-RNN & Transformer training scripts
│   ├── results/                # Quantitative benchmark metrics & training logs
│   └── work/                   # Fast fine-tuning & evaluation scripts
├── .gitignore                  # Prevents raw datasets & heavy >100MB files
└── README.md                   # Project documentation
```

---

## 🛠️ Model Weights Note

- **ResNet-50 + GRU (`gru_best.pt`, 29.8 MB)** is bundled directly inside `webapp/checkpoints/` so the app works immediately out of the box.
- **Salesforce BLIP** and **ViT-GPT-2** base models are loaded automatically via Hugging Face `transformers` on first run without needing manual file downloads.
