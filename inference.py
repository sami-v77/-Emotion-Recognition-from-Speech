"""
Inference & Visualization Module for Speech Emotion Recognition

- Real-time prediction from microphone or audio file
- Probability visualization (bar chart)
- Confusion matrix plotting
- Training curve plotting
- Grad-CAM style saliency for CNN models
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from feature_extraction import AudioFeatureExtractor


# ─── Emotion Color Palette ────────────────────────────────────────────────────

EMOTION_COLORS = {
    "neutral":   "#7f8c8d",
    "happy":     "#f1c40f",
    "sad":       "#3498db",
    "angry":     "#e74c3c",
    "fearful":   "#9b59b6",
    "disgust":   "#27ae60",
    "surprised": "#e67e22",
}

EMOTION_EMOJIS = {
    "neutral": "😐", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}


# ─── Inference Engine ─────────────────────────────────────────────────────────

class EmotionPredictor:
    """
    Real-time inference wrapper around a trained SER model.

    Args:
        model: Trained nn.Module
        class_names: Ordered list of emotion class labels
        extractor: AudioFeatureExtractor instance
        device: torch.device
        input_format: 'cnn' → (1, n_feat, T), 'lstm' → (1, T, n_feat)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        extractor: AudioFeatureExtractor,
        device: torch.device,
        input_format: str = "cnn",  # 'cnn' or 'lstm'
    ):
        self.model = model.eval().to(device)
        self.class_names = class_names
        self.extractor = extractor
        self.device = device
        self.input_format = input_format

    def _prepare_input(self, file_path: str) -> Optional[torch.Tensor]:
        """Extract features and format as model input tensor."""
        features = self.extractor.extract_all_features(file_path)
        if features is None:
            return None
        x = torch.from_numpy(features).float()  # (n_feat, T)
        if self.input_format == "lstm":
            x = x.T  # (T, n_feat)
        elif self.input_format == "cnn2d":
            x = x.unsqueeze(0)  # (1, n_feat, T)
        x = x.unsqueeze(0).to(self.device)  # add batch dim
        return x

    @torch.no_grad()
    def predict(self, file_path: str) -> Dict:
        """
        Predict emotion from an audio file.

        Returns:
            {
                "emotion": str,        # predicted emotion label
                "confidence": float,  # probability of top class
                "probabilities": dict, # {emotion: probability}
                "emoji": str,          # emotion emoji
            }
        """
        x = self._prepare_input(file_path)
        if x is None:
            return {"error": "Feature extraction failed"}

        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        top_idx = int(np.argmax(probs))
        emotion = self.class_names[top_idx]

        result = {
            "emotion": emotion,
            "confidence": float(probs[top_idx]),
            "probabilities": {
                self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
            },
            "emoji": EMOTION_EMOJIS.get(emotion, ""),
        }
        return result

    def predict_batch(self, file_paths: List[str]) -> List[Dict]:
        """Predict emotions for a list of audio files."""
        return [self.predict(fp) for fp in file_paths]

    def predict_from_microphone(self, duration: float = 3.0, sr: int = 22050) -> Dict:
        """
        Record from microphone and predict emotion.
        Requires: sounddevice, scipy
        """
        try:
            import sounddevice as sd
            from scipy.io import wavfile
            import tempfile

            print(f"🎙️  Recording {duration}s of audio...")
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wavfile.write(tmp.name, sr, audio)
                result = self.predict(tmp.name)
                os.unlink(tmp.name)

            print(f"✓ Detected emotion: {result['emotion']} {result['emoji']} "
                  f"(confidence: {result['confidence']:.1%})")
            return result

        except ImportError:
            return {"error": "sounddevice and scipy required for microphone recording. "
                             "Install: pip install sounddevice scipy"}


# ─── Visualization ────────────────────────────────────────────────────────────

class SERVisualizer:
    """Plotting utilities for SER results."""

    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

    def plot_emotion_probabilities(
        self, result: Dict, title: str = "Emotion Probabilities", save: bool = True
    ):
        """Horizontal bar chart of per-emotion probabilities."""
        probs = result["probabilities"]
        emotions = list(probs.keys())
        values = [probs[e] for e in emotions]
        colors = [EMOTION_COLORS.get(e, "#95a5a6") for e in emotions]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(emotions, values, color=colors, edgecolor="white", height=0.6)

        for bar, val in zip(bars, values):
            ax.text(
                val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9, color="#2c3e50"
            )

        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Probability", fontsize=11)
        ax.set_title(
            f"{title}\nPredicted: {result['emotion'].upper()} {result.get('emoji', '')} "
            f"({result['confidence']:.1%})",
            fontsize=12, fontweight="bold", pad=12
        )
        ax.axvline(x=0.5, linestyle="--", color="gray", alpha=0.4, lw=0.8)
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "emotion_probabilities.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        return fig

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = True,
        save: bool = True,
    ):
        """Annotated confusion matrix heatmap."""
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt, vmax = ".2f", 1.0
        else:
            fmt, vmax = "d", cm.max()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=0,
            vmax=vmax,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "confusion_matrix.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        return fig

    def plot_training_curves(
        self, history_path: str, save: bool = True
    ):
        """Loss and accuracy curves from training history JSON."""
        with open(history_path) as f:
            history = json.load(f)

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Training History", fontsize=14, fontweight="bold")

        # Loss
        axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c", lw=2)
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#3498db", lw=2, linestyle="--")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Cross-Entropy Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#e74c3c", lw=2)
        axes[1].plot(epochs, history["val_acc"], label="Val Acc", color="#3498db", lw=2, linestyle="--")
        if "val_f1" in history:
            axes[1].plot(epochs, history["val_f1"], label="Val F1", color="#27ae60", lw=2, linestyle=":")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Accuracy & F1")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "training_curves.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        return fig

    def plot_mfcc(
        self,
        file_path: str,
        extractor: AudioFeatureExtractor,
        title: str = "MFCC Features",
        save: bool = True,
    ):
        """Visualize MFCCs of an audio file."""
        import librosa
        import librosa.display

        audio = extractor.load_audio(file_path)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=extractor.sample_rate, n_mfcc=extractor.n_mfcc
        )

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # Waveform
        times = np.linspace(0, len(audio) / extractor.sample_rate, len(audio))
        axes[0].plot(times, audio, color="#2980b9", lw=0.6)
        axes[0].set_title("Waveform", fontsize=11)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(alpha=0.2)

        # MFCC
        img = librosa.display.specshow(
            mfcc,
            sr=extractor.sample_rate,
            hop_length=extractor.hop_length,
            x_axis="time",
            ax=axes[1],
            cmap="magma",
        )
        fig.colorbar(img, ax=axes[1], label="Magnitude (dB)")
        axes[1].set_title(f"MFCC ({extractor.n_mfcc} coefficients)", fontsize=11)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "mfcc_visualization.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        return fig

    def plot_class_distribution(
        self, emotion_counts: Dict[str, int], save: bool = True
    ):
        """Bar chart of emotion class distribution in the dataset."""
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = [EMOTION_COLORS.get(e, "#95a5a6") for e in emotions]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(emotions, counts, color=colors, edgecolor="white", width=0.6)
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(cnt), ha="center", va="bottom", fontsize=9
            )
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Sample Count")
        ax.set_title("Dataset Class Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "class_distribution.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig


if __name__ == "__main__":
    # Demo: run from project root
    viz = SERVisualizer(output_dir="results/plots")

    # Simulate a prediction result for demo
    demo_result = {
        "emotion": "happy",
        "confidence": 0.82,
        "emoji": "😄",
        "probabilities": {
            "neutral": 0.04, "happy": 0.82, "sad": 0.02,
            "angry": 0.05, "fearful": 0.03, "disgust": 0.02, "surprised": 0.02
        }
    }
    viz.plot_emotion_probabilities(demo_result, save=False)
    plt.show()
