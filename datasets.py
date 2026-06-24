"""
Dataset Loaders for Emotion Recognition
Supports: RAVDESS, TESS, EMO-DB
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from feature_extraction import AudioFeatureExtractor


# ─── Emotion Mappings ──────────────────────────────────────────────────────────

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "pleasant_surprise": "surprised",
    "sad": "sad",
}

EMODB_EMOTION_MAP = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

# Unified 7-class emotion set
UNIFIED_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


# ─── Base Dataset Class ────────────────────────────────────────────────────────

class EmotionDataset:
    """Base class for emotion speech datasets."""

    def __init__(self, root_dir: str, extractor: AudioFeatureExtractor):
        self.root_dir = Path(root_dir)
        self.extractor = extractor
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(UNIFIED_EMOTIONS)

    def _get_file_list(self) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def build_dataframe(self) -> pd.DataFrame:
        """Scan dataset directory and return a DataFrame with file paths and labels."""
        records = self._get_file_list()
        df = pd.DataFrame(records, columns=["file_path", "emotion"])
        df = df[df["emotion"].isin(UNIFIED_EMOTIONS)].reset_index(drop=True)
        print(f"[{self.__class__.__name__}] Found {len(df)} samples | Emotions: {df['emotion'].value_counts().to_dict()}")
        return df

    def extract_features(
        self,
        df: pd.DataFrame,
        feature_type: str = "deep",  # 'deep' for CNN/LSTM or 'statistical' for SVM
        cache_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for all audio files in the dataframe.
        
        Args:
            df: DataFrame with 'file_path' and 'emotion' columns
            feature_type: 'deep' returns 2D arrays; 'statistical' returns 1D vectors
            cache_path: If provided, save/load features from .npz cache

        Returns:
            X: Feature array
            y: Encoded integer labels
        """
        if cache_path and os.path.exists(cache_path):
            print(f"[CACHE] Loading features from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return data["X"], data["y"]

        X, y = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            if feature_type == "deep":
                feat = self.extractor.extract_all_features(row["file_path"])
            else:
                feat = self.extractor.extract_statistical_features(row["file_path"])

            if feat is not None:
                X.append(feat)
                y.append(row["emotion"])

        X = np.array(X, dtype=np.float32)
        y = self.label_encoder.transform(y)

        if cache_path:
            np.savez(cache_path, X=X, y=y)
            print(f"[CACHE] Saved features to {cache_path}")

        return X, y

    def get_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple:
        """Split dataset into train / validation / test sets (stratified)."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=random_state, stratify=y_train
        )
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test


# ─── RAVDESS Loader ────────────────────────────────────────────────────────────

class RAVDESSDataset(EmotionDataset):
    """
    RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
    
    File naming: 03-01-06-01-02-01-12.wav
    Position 3 (index 2) → emotion code
    
    Download: https://zenodo.org/record/1188976
    Unzip so that: <root_dir>/Actor_01/*.wav, Actor_02/*.wav, ...
    """

    def _get_file_list(self) -> List[Tuple[str, str]]:
        records = []
        for wav_file in sorted(self.root_dir.rglob("*.wav")):
            parts = wav_file.stem.split("-")
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
            if emotion:
                records.append((str(wav_file), emotion))
        return records


# ─── TESS Loader ──────────────────────────────────────────────────────────────

class TESSDataset(EmotionDataset):
    """
    TESS: Toronto Emotional Speech Set
    
    Folder structure: <root_dir>/OAF_angry/*.wav, YAF_happy/*.wav, ...
    Emotion is encoded in folder name after underscore.
    
    Download: https://tspace.library.utoronto.ca/handle/1807/24487
    """

    def _get_file_list(self) -> List[Tuple[str, str]]:
        records = []
        for wav_file in sorted(self.root_dir.rglob("*.wav")):
            folder = wav_file.parent.name.lower()
            # e.g., "OAF_angry" → "angry"
            parts = folder.split("_", 1)
            if len(parts) < 2:
                continue
            raw_emotion = parts[1]
            emotion = TESS_EMOTION_MAP.get(raw_emotion)
            if emotion:
                records.append((str(wav_file), emotion))
        return records


# ─── EMO-DB Loader ────────────────────────────────────────────────────────────

class EMODBDataset(EmotionDataset):
    """
    EMO-DB: Berlin Database of Emotional Speech
    
    File naming: 03a01Fa.wav
    Character at index 5 → emotion code
    
    Download: http://emodb.bilderbar.info/download/
    Unzip .wav files into <root_dir>/*.wav
    """

    def _get_file_list(self) -> List[Tuple[str, str]]:
        records = []
        for wav_file in sorted(self.root_dir.glob("*.wav")):
            name = wav_file.stem
            if len(name) < 6:
                continue
            emotion_code = name[5].upper()
            emotion = EMODB_EMOTION_MAP.get(emotion_code)
            if emotion:
                records.append((str(wav_file), emotion))
        return records


# ─── Combined Dataset ─────────────────────────────────────────────────────────

class CombinedDataset:
    """Merge multiple dataset loaders into one unified dataset."""

    def __init__(self, datasets: List[EmotionDataset]):
        self.datasets = datasets

    def build_dataframe(self) -> pd.DataFrame:
        frames = [ds.build_dataframe() for ds in self.datasets]
        combined = pd.concat(frames, ignore_index=True)
        print(f"\n[Combined] Total samples: {len(combined)}")
        print(combined["emotion"].value_counts())
        return combined


# ─── Demo / Quick Start ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example usage (replace paths with your actual dataset directories)
    extractor = AudioFeatureExtractor(sample_rate=22050, n_mfcc=40, duration=3.0)

    # RAVDESS
    ravdess = RAVDESSDataset(root_dir="data/RAVDESS", extractor=extractor)
    df_ravdess = ravdess.build_dataframe()

    # TESS
    tess = TESSDataset(root_dir="data/TESS", extractor=extractor)
    df_tess = tess.build_dataframe()

    # Combine
    combined_df = pd.concat([df_ravdess, df_tess], ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} samples")
    print(combined_df["emotion"].value_counts())
