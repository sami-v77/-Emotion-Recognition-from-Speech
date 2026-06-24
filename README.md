# 🎤 Speech Emotion Recognition (SER)
> Recognize human emotions from speech using deep learning and signal processing.

---

## 📋 Overview

This project implements a complete, modular pipeline for **Speech Emotion Recognition (SER)** — detecting emotions like *happy*, *sad*, *angry*, *fearful*, *neutral*, *disgusted*, and *surprised* from raw audio.

| Component | Details |
|---|---|
| **Features** | MFCCs (+ Δ, ΔΔ), Chroma, Mel Spectrogram, Spectral Centroid/Bandwidth/Rolloff, ZCR, RMS |
| **Models** | 1D CNN, 2D CNN, BiLSTM, CNN-LSTM, Attention-LSTM, Transformer Encoder |
| **Datasets** | RAVDESS, TESS, EMO-DB (or combined) |
| **Training** | Mixed-precision AMP, class-weighted loss, label smoothing, early stopping |
| **Inference** | File-based or real-time microphone |

---

## 🏗️ Project Structure

```
emotion_recognition/
├── src/
│   ├── feature_extraction.py   # MFCC, Chroma, Mel Spec, ZCR, RMS extraction
│   ├── datasets.py             # RAVDESS, TESS, EMO-DB loaders
│   ├── models.py               # CNN, LSTM, Transformer architectures
│   ├── trainer.py              # Training loop, early stopping, AMP
│   ├── inference.py            # Predictor + visualization utilities
│   └── main.py                 # End-to-end CLI pipeline
├── data/
│   ├── RAVDESS/                # → place dataset here
│   ├── TESS/                   # → place dataset here
│   ├── EMODB/                  # → place dataset here
│   └── cache/                  # auto-generated feature cache (.npz)
├── results/
│   ├── best_model.pth          # auto-saved best checkpoint
│   ├── training_log.csv        # per-epoch metrics
│   ├── history.json            # training history
│   ├── test_results.json       # final test metrics
│   └── plots/                  # all generated figures
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download a dataset

**RAVDESS** (most popular, ~1500 samples, 24 actors):
```
https://zenodo.org/record/1188976
```
Unzip so the structure is: `data/RAVDESS/Actor_01/*.wav`, `data/RAVDESS/Actor_02/*.wav`, ...

**TESS** (~2800 samples):
```
https://tspace.library.utoronto.ca/handle/1807/24487
```
Unzip so the structure is: `data/TESS/OAF_angry/*.wav`, `data/TESS/YAF_happy/*.wav`, ...

**EMO-DB** (German speech, ~535 samples):
```
http://emodb.bilderbar.info/download/
```
Unzip `.wav` files directly into `data/EMODB/*.wav`

### 3. Train a model
```bash
cd src

# CNN-LSTM on RAVDESS (recommended starting point)
python main.py train --dataset ravdess --data_dir ../data/RAVDESS --model cnn_lstm

# Transformer on TESS
python main.py train --dataset tess --data_dir ../data/TESS --model transformer

# BiLSTM on EMO-DB
python main.py train --dataset emodb --data_dir ../data/EMODB --model bilstm
```

### 4. Run inference
```bash
# From a .wav file
python main.py infer --audio path/to/speech.wav \
                     --checkpoint ../results/best_model.pth \
                     --model cnn_lstm
```

---

## 🧠 Feature Extraction

All features are extracted using `librosa` and concatenated into a single tensor:

| Feature | Shape | Description |
|---|---|---|
| MFCC + Δ + ΔΔ | (120, T) | Captures vocal tract shape and its dynamics |
| Chroma | (12, T) | Pitch class distribution (tonal content) |
| Mel Spectrogram | (128, T) | Frequency content on perceptual mel scale |
| Spectral Centroid/BW/Rolloff/Contrast | (9, T) | Spectral shape descriptors |
| ZCR + RMS | (2, T) | Noisiness and energy |
| **Total** | **(271, T)** | Combined feature tensor |

---

## 🏛️ Model Architectures

### 1. CNN-LSTM (Recommended)
```
Input (B, 271, T)
  └→ Conv1D stack (128→256→256 channels, MaxPool, Dropout)
  └→ BiLSTM (hidden=256, layers=2, bidirectional)
  └→ Self-Attention (temporal pooling)
  └→ Linear (256→128→n_classes)
```

### 2. Transformer Encoder
```
Input (B, T, 271)
  └→ Linear projection → d_model=256
  └→ Positional Encoding (sinusoidal)
  └→ 4× TransformerEncoderLayer (Pre-LN, nhead=8)
  └→ Global Average Pool over time
  └→ Linear (256→128→n_classes)
```

### 3. Bidirectional LSTM
```
Input (B, T, 271)
  └→ BiLSTM (hidden=256, layers=3, bidirectional)
  └→ Concat final hidden states (fwd + bwd)
  └→ Linear (512→128→n_classes)
```

---

## ⚙️ Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| LR Schedule | CosineAnnealingWarmRestarts (T₀=20) |
| Loss | CrossEntropy + Class Weights + Label Smoothing (0.1) |
| Augmentation | Time masking (10%) + Gaussian noise |
| Sampler | Weighted random sampling for class balance |
| Mixed Precision | AMP (torch.cuda.amp) on CUDA |
| Early Stopping | Patience = 15 epochs on val accuracy |
| Gradient Clipping | max_norm = 1.0 |

---

## 📊 Expected Results

| Dataset | Model | Val Acc | Test Acc |
|---|---|---|---|
| RAVDESS | CNN-LSTM | ~78–83% | ~75–80% |
| TESS | Transformer | ~88–93% | ~86–91% |
| RAVDESS+TESS | CNN-LSTM | ~82–87% | ~80–85% |
| EMO-DB | BiLSTM | ~80–85% | ~78–82% |

> Results vary by random seed, hyperparameter tuning, and preprocessing choices.

---

## 📈 Outputs

After training, `results/` will contain:
- `best_model.pth` — best checkpoint (by val accuracy)
- `training_log.csv` — per-epoch metrics table
- `history.json` — loss/acc/f1 curves
- `test_results.json` — final accuracy and F1
- `plots/training_curves.png` — loss & accuracy curves
- `plots/confusion_matrix.png` — normalized confusion matrix

---

## 🔬 Extending the Project

**Add a new dataset:** Subclass `EmotionDataset` in `datasets.py` and implement `_get_file_list()`.

**Add a new model:** Implement a new `nn.Module` in `models.py` and register it in `build_model()`.

**Hyperparameter search:** Wrap `SERTrainer.train()` with `optuna` or `ray[tune]` for automated tuning.

**Data augmentation:** Extend `EmotionSpeechDataset._augment()` in `trainer.py` — add pitch shifting, time stretching, or SpecAugment.

---

## 📚 References

- RAVDESS: Livingstone & Russo (2018) — https://doi.org/10.1371/journal.pone.0196391  
- TESS: Pichora-Fuller & Dupuis (2020) — University of Toronto  
- EMO-DB: Burkhardt et al. (2005) — TU Berlin  
- librosa: McFee et al. (2015) — https://librosa.org  
- SpecAugment: Park et al. (2019) — https://arxiv.org/abs/1904.08779
