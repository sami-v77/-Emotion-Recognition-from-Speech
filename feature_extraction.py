"""
Feature Extraction Module for Emotion Recognition from Speech
Extracts MFCCs, Chroma, Mel Spectrogram, ZCR, and other audio features.
"""

import numpy as np
import librosa
import librosa.display
import os
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """
    Extracts rich acoustic features from speech audio files.
    
    Features extracted:
    - MFCC (Mel-Frequency Cepstral Coefficients) — captures timbral texture
    - Chroma Features — captures pitch class information
    - Mel Spectrogram — frequency representation on mel scale
    - Zero Crossing Rate — captures noisiness / voicing
    - RMS Energy — captures loudness dynamics
    - Spectral Centroid, Bandwidth, Rolloff — spectral shape
    - Pitch (F0) — fundamental frequency contour
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 40,
        n_chroma: int = 12,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048,
        duration: float = 3.0,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.duration = duration
        self.max_pad_len = int(duration * sample_rate / hop_length) + 1

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        audio, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            duration=self.duration,
            mono=True
        )
        # Normalize amplitude
        audio = librosa.util.normalize(audio)
        return audio

    def pad_or_truncate(self, feature: np.ndarray) -> np.ndarray:
        """Pad or truncate feature array to fixed length."""
        if feature.shape[1] < self.max_pad_len:
            pad_width = self.max_pad_len - feature.shape[1]
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        else:
            feature = feature[:, :self.max_pad_len]
        return feature

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features with delta and delta-delta."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # Shape: (3*n_mfcc, T)

    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract Chroma features (pitch class profiles)."""
        stft = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        chroma = librosa.feature.chroma_stft(S=stft, sr=self.sample_rate, n_chroma=self.n_chroma)
        return chroma  # Shape: (12, T)

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract Mel Spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db  # Shape: (n_mels, T)

    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral shape features."""
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
        return np.vstack([centroid, bandwidth, rolloff, contrast])  # Shape: (9, T)

    def extract_zcr_rms(self, audio: np.ndarray) -> np.ndarray:
        """Extract Zero Crossing Rate and RMS Energy."""
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
        return np.vstack([zcr, rms])  # Shape: (2, T)

    def extract_all_features(self, file_path: str) -> np.ndarray | None:
        """
        Extract and concatenate all features into a unified tensor.
        Returns shape: (total_features, time_steps)
        """
        try:
            audio = self.load_audio(file_path)

            mfcc = self.pad_or_truncate(self.extract_mfcc(audio))
            chroma = self.pad_or_truncate(self.extract_chroma(audio))
            mel = self.pad_or_truncate(self.extract_mel_spectrogram(audio))
            spectral = self.pad_or_truncate(self.extract_spectral_features(audio))
            zcr_rms = self.pad_or_truncate(self.extract_zcr_rms(audio))

            # Concatenate along feature axis
            combined = np.vstack([mfcc, chroma, mel, spectral, zcr_rms])
            return combined  # Shape: (3*40 + 12 + 128 + 9 + 2, T) = (271, T)

        except Exception as e:
            print(f"[ERROR] Feature extraction failed for {file_path}: {e}")
            return None

    def extract_statistical_features(self, file_path: str) -> np.ndarray | None:
        """
        Extract statistical summaries (mean, std, min, max, skew) for each feature.
        Returns 1D vector suitable for traditional ML classifiers.
        """
        try:
            from scipy.stats import skew, kurtosis
            audio = self.load_audio(file_path)
            features = []

            mfcc = self.extract_mfcc(audio)
            for stat_fn in [np.mean, np.std, lambda x: skew(x, axis=1)]:
                features.append(stat_fn(mfcc) if callable(stat_fn) else stat_fn(mfcc, axis=1))

            chroma = self.extract_chroma(audio)
            features.append(np.mean(chroma, axis=1))
            features.append(np.std(chroma, axis=1))

            mel = self.extract_mel_spectrogram(audio)
            features.append(np.mean(mel, axis=1))
            features.append(np.std(mel, axis=1))

            spectral = self.extract_spectral_features(audio)
            features.append(np.mean(spectral, axis=1))
            features.append(np.std(spectral, axis=1))

            zcr_rms = self.extract_zcr_rms(audio)
            features.append(np.mean(zcr_rms, axis=1))

            return np.concatenate(features)

        except Exception as e:
            print(f"[ERROR] Statistical feature extraction failed: {e}")
            return None
