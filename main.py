"""
Main Pipeline: End-to-End Speech Emotion Recognition
=====================================================

Usage:
    python main.py --dataset ravdess --data_dir data/RAVDESS --model cnn_lstm
    python main.py --dataset tess    --data_dir data/TESS    --model transformer
    python main.py --infer path/to/audio.wav --checkpoint results/best_model.pth

Steps:
    1. Load dataset → build file manifest
    2. Extract acoustic features (MFCCs, Chroma, Mel Spec, ...)
    3. Train / validate / test selected model
    4. Plot confusion matrix & training curves
    5. (Optional) run real-time inference
"""

import os
import argparse
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from feature_extraction import AudioFeatureExtractor
from datasets import RAVDESSDataset, TESSDataset, EMODBDataset, UNIFIED_EMOTIONS
from models import build_model
from trainer import SERTrainer
from inference import EmotionPredictor, SERVisualizer


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Audio
    "sample_rate": 22050,
    "n_mfcc": 40,
    "n_mels": 128,
    "duration": 3.0,
    "hop_length": 512,
    "n_fft": 2048,

    # Model
    "model": "cnn_lstm",      # cnn1d | cnn2d | bilstm | cnn_lstm | attention_lstm | transformer
    "n_classes": 7,

    # Training
    "epochs": 100,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 15,
    "label_smoothing": 0.1,
    "use_amp": True,

    # Paths
    "output_dir": "results",
    "cache_dir": "data/cache",
}


# ─── Input format per model ───────────────────────────────────────────────────

MODEL_INPUT_FORMAT = {
    "cnn1d": "cnn",
    "cnn2d": "cnn2d",
    "bilstm": "lstm",
    "cnn_lstm": "cnn",
    "attention_lstm": "lstm",
    "transformer": "lstm",
}


# ─── Dataset loader factory ───────────────────────────────────────────────────

def load_dataset(dataset_name: str, data_dir: str, extractor: AudioFeatureExtractor):
    loaders = {
        "ravdess": RAVDESSDataset,
        "tess": TESSDataset,
        "emodb": EMODBDataset,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")
    return loaders[dataset_name](root_dir=data_dir, extractor=extractor)


# ─── Feature reshape for model input ─────────────────────────────────────────

def reshape_features(X: np.ndarray, input_format: str) -> np.ndarray:
    """
    X shape coming in: (N, n_features, T)
    
    - cnn1d / cnn_lstm → keep (N, n_features, T)
    - cnn2d            → expand (N, 1, n_features, T)
    - lstm / transformer → transpose (N, T, n_features)
    """
    if input_format == "cnn":
        return X
    elif input_format == "cnn2d":
        return X[:, np.newaxis, :, :]  # (N, 1, F, T)
    elif input_format == "lstm":
        return X.transpose(0, 2, 1)   # (N, T, F)
    return X


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_training(args):
    cfg = DEFAULT_CONFIG.copy()
    cfg["model"] = args.model
    cfg["output_dir"] = args.output_dir

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\n[Device] {device}")

    # ── Step 1: Feature Extractor ──────────────────────────────────────────────
    extractor = AudioFeatureExtractor(
        sample_rate=cfg["sample_rate"],
        n_mfcc=cfg["n_mfcc"],
        n_mels=cfg["n_mels"],
        duration=cfg["duration"],
        hop_length=cfg["hop_length"],
        n_fft=cfg["n_fft"],
    )

    # ── Step 2: Load Dataset ──────────────────────────────────────────────────
    print(f"\n[Dataset] Loading {args.dataset.upper()} from {args.data_dir}")
    dataset = load_dataset(args.dataset, args.data_dir, extractor)
    df = dataset.build_dataframe()

    # ── Step 3: Extract Features ──────────────────────────────────────────────
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    cache_path = os.path.join(cfg["cache_dir"], f"{args.dataset}_features.npz")

    print("\n[Features] Extracting acoustic features...")
    X, y = dataset.extract_features(df, feature_type="deep", cache_path=cache_path)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")

    n_features = X.shape[1]  # number of acoustic feature channels

    # ── Step 4: Train/Val/Test Split ──────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_splits(X, y)

    # Reshape for model input format
    fmt = MODEL_INPUT_FORMAT[args.model]
    X_train = reshape_features(X_train, fmt)
    X_val   = reshape_features(X_val,   fmt)
    X_test  = reshape_features(X_test,  fmt)

    input_dim = X_train.shape[-1] if fmt in ("lstm", "transformer") else n_features
    print(f"  Input format: {fmt} | Input dim: {input_dim}")

    # ── Step 5: Build Model ───────────────────────────────────────────────────
    print(f"\n[Model] Building {args.model.upper()}...")
    model = build_model(
        model_name=args.model,
        n_features=input_dim,
        n_classes=cfg["n_classes"],
    )

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    trainer = SERTrainer(
        model=model,
        device=device,
        output_dir=cfg["output_dir"],
        class_names=UNIFIED_EMOTIONS,
    )

    train_loader, val_loader, test_loader = trainer.build_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=cfg["batch_size"],
        use_weighted_sampler=True,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        y_train=y_train,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience"],
        use_amp=cfg["use_amp"],
        label_smoothing=cfg["label_smoothing"],
    )

    # ── Step 7: Evaluate ──────────────────────────────────────────────────────
    trainer.load_best()
    results = trainer.evaluate(test_loader)

    # ── Step 8: Visualize ─────────────────────────────────────────────────────
    viz = SERVisualizer(output_dir=os.path.join(cfg["output_dir"], "plots"))

    viz.plot_training_curves(
        os.path.join(cfg["output_dir"], "history.json")
    )
    viz.plot_confusion_matrix(
        np.array(results["confusion_matrix"]),
        class_names=UNIFIED_EMOTIONS,
        title=f"Confusion Matrix — {args.model.upper()}",
    )

    print(f"\n✅ Pipeline complete. Outputs saved to: {cfg['output_dir']}/")


def run_inference(args):
    """Single-file inference using a pre-trained checkpoint."""
    device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt.get("class_names", UNIFIED_EMOTIONS)

    extractor = AudioFeatureExtractor()
    model = build_model(
        model_name=args.model,
        n_features=271,  # default combined feature size
        n_classes=len(class_names),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    predictor = EmotionPredictor(
        model=model,
        class_names=class_names,
        extractor=extractor,
        device=device,
        input_format=MODEL_INPUT_FORMAT[args.model],
    )

    result = predictor.predict(args.infer)
    print(f"\n🎯 Detected Emotion: {result['emotion'].upper()} {result.get('emoji', '')}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print("\n   Probabilities:")
    for emotion, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"   {emotion:12s} {bar:30s} {prob:.1%}")

    viz = SERVisualizer(output_dir="results/plots")
    viz.plot_emotion_probabilities(result)
    plt.show()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--dataset", type=str, default="ravdess",
                               choices=["ravdess", "tess", "emodb"],
                               help="Dataset to train on")
    train_parser.add_argument("--data_dir", type=str, required=True,
                               help="Root directory of the dataset")
    train_parser.add_argument("--model", type=str, default="cnn_lstm",
                               choices=list(MODEL_INPUT_FORMAT.keys()),
                               help="Model architecture")
    train_parser.add_argument("--output_dir", type=str, default="results",
                               help="Output directory for checkpoints and logs")

    # Infer subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference on an audio file")
    infer_parser.add_argument("--audio", type=str, required=True,
                               help="Path to audio file (.wav)")
    infer_parser.add_argument("--checkpoint", type=str, required=True,
                               help="Path to model checkpoint (.pth)")
    infer_parser.add_argument("--model", type=str, default="cnn_lstm",
                               choices=list(MODEL_INPUT_FORMAT.keys()))

    return parser.parse_args()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    args = parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "infer":
        # Adapt namespace for run_inference
        args.infer = args.audio
        run_inference(args)
    else:
        print("Usage:")
        print("  python main.py train --dataset ravdess --data_dir data/RAVDESS --model cnn_lstm")
        print("  python main.py infer --audio sample.wav --checkpoint results/best_model.pth")
