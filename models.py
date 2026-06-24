"""
Deep Learning Model Architectures for Speech Emotion Recognition

Models implemented:
1. CNN1D       — 1D Convolutional Network on feature sequences
2. CNN2D       — 2D Convolutional Network treating features as spectrograms
3. BiLSTM      — Bidirectional LSTM for sequential modeling
4. CNN_LSTM    — CNN feature extractor + LSTM temporal model (hybrid)
5. AttentionLSTM — LSTM with temporal self-attention mechanism
6. TransformerSER — Transformer encoder for SER (modern SOTA direction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Utility Blocks ────────────────────────────────────────────────────────────

class ConvBlock1D(nn.Module):
    """1D Conv → BN → ReLU → Dropout → MaxPool"""
    def __init__(self, in_ch, out_ch, kernel=3, pool=2, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock2D(nn.Module):
    """2D Conv → BN → ReLU → Dropout → MaxPool"""
    def __init__(self, in_ch, out_ch, kernel=(3, 3), pool=(2, 2), dropout=0.3):
        super().__init__()
        pad = (kernel[0] // 2, kernel[1] // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    """Additive self-attention over time dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (B, T, H)
        scores = self.attention(lstm_out)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)    # (B, T, 1)
        context = (weights * lstm_out).sum(dim=1) # (B, H)
        return context


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─── Model 1: 1D CNN ──────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    1D CNN treating feature channels as input channels, time as sequence.
    Input: (B, n_features, T)
    """
    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.4):
        super().__init__()
        self.conv_stack = nn.Sequential(
            ConvBlock1D(n_features, 128, kernel=3, pool=2, dropout=dropout),
            ConvBlock1D(128, 256, kernel=3, pool=2, dropout=dropout),
            ConvBlock1D(256, 512, kernel=3, pool=2, dropout=dropout),
            ConvBlock1D(512, 256, kernel=3, pool=2, dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        return self.classifier(x)


# ─── Model 2: 2D CNN ──────────────────────────────────────────────────────────

class CNN2D(nn.Module):
    """
    2D CNN treating (features × time) as an image (like a spectrogram).
    Input: (B, 1, n_features, T)
    """
    def __init__(self, n_classes: int, dropout: float = 0.4):
        super().__init__()
        self.conv_stack = nn.Sequential(
            ConvBlock2D(1, 32, kernel=(3, 3), pool=(2, 2), dropout=dropout),
            ConvBlock2D(32, 64, kernel=(3, 3), pool=(2, 2), dropout=dropout),
            ConvBlock2D(64, 128, kernel=(3, 3), pool=(2, 2), dropout=dropout),
            ConvBlock2D(128, 256, kernel=(3, 3), pool=(2, 2), dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        return self.classifier(x)


# ─── Model 3: Bidirectional LSTM ──────────────────────────────────────────────

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for capturing temporal emotion dynamics.
    Input: (B, T, n_features)
    """
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        # Concatenate final hidden states from both directions
        hn_fwd = hn[-2, :, :]  # Last layer, forward
        hn_bwd = hn[-1, :, :]  # Last layer, backward
        context = torch.cat([hn_fwd, hn_bwd], dim=1)
        return self.classifier(context)


# ─── Model 4: CNN + LSTM Hybrid ───────────────────────────────────────────────

class CNN_LSTM(nn.Module):
    """
    CNN-LSTM hybrid: CNN extracts local patterns, LSTM captures temporal dynamics.
    Input: (B, n_features, T)
    """
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        cnn_channels: int = 128,
        lstm_hidden: int = 256,
        n_lstm_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            ConvBlock1D(n_features, cnn_channels, kernel=3, pool=2, dropout=dropout),
            ConvBlock1D(cnn_channels, cnn_channels * 2, kernel=3, pool=2, dropout=dropout),
            ConvBlock1D(cnn_channels * 2, cnn_channels * 2, kernel=3, pool=1, dropout=dropout),
        )
        cnn_out_channels = cnn_channels * 2

        # LSTM temporal model
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.attention = SelfAttention(lstm_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (B, n_features, T)
        x = self.cnn(x)            # (B, cnn_ch, T')
        x = x.permute(0, 2, 1)    # (B, T', cnn_ch)
        x, _ = self.lstm(x)       # (B, T', 2*lstm_h)
        x = self.attention(x)     # (B, 2*lstm_h)
        return self.classifier(x)


# ─── Model 5: Attention LSTM ──────────────────────────────────────────────────

class AttentionLSTM(nn.Module):
    """
    BiLSTM with temporal self-attention.
    Input: (B, T, n_features)
    """
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = SelfAttention(hidden_dim * 2)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)            # (B, T, 2H)
        lstm_out = self.layer_norm(lstm_out)  # stabilize
        context = self.attention(lstm_out)    # (B, 2H)
        return self.classifier(context)


# ─── Model 6: Transformer Encoder ─────────────────────────────────────────────

class TransformerSER(nn.Module):
    """
    Transformer Encoder for Speech Emotion Recognition.
    Modern SOTA direction; captures global temporal dependencies.
    Input: (B, T, n_features)
    """
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        n_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_encoding(x)        # add positional info
        x = self.transformer(x)         # (B, T, d_model)
        x = x.mean(dim=1)              # global average pool over time
        return self.classifier(x)


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    n_features: int,
    n_classes: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to build a model by name.

    Args:
        model_name: One of ['cnn1d', 'cnn2d', 'bilstm', 'cnn_lstm', 'attention_lstm', 'transformer']
        n_features: Number of acoustic feature channels
        n_classes: Number of emotion classes
        **kwargs: Additional model hyperparameters

    Returns:
        Initialized nn.Module
    """
    model_name = model_name.lower()
    models = {
        "cnn1d": lambda: CNN1D(n_features, n_classes, **kwargs),
        "cnn2d": lambda: CNN2D(n_classes, **kwargs),
        "bilstm": lambda: BiLSTM(n_features, n_classes, **kwargs),
        "cnn_lstm": lambda: CNN_LSTM(n_features, n_classes, **kwargs),
        "attention_lstm": lambda: AttentionLSTM(n_features, n_classes, **kwargs),
        "transformer": lambda: TransformerSER(n_features, n_classes, **kwargs),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    model = models[model_name]()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name.upper()} | Parameters: {n_params:,}")
    return model


if __name__ == "__main__":
    # Quick sanity check on all models
    B, n_feat, T, n_cls = 4, 271, 130, 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (B, n_feat, T) input for CNN-style models
    x_cnn = torch.randn(B, n_feat, T).to(device)
    # (B, T, n_feat) input for LSTM/Transformer
    x_seq = torch.randn(B, T, n_feat).to(device)
    # (B, 1, n_feat, T) input for 2D CNN
    x_2d = torch.randn(B, 1, n_feat, T).to(device)

    test_cases = [
        ("cnn1d", x_cnn),
        ("cnn2d", x_2d),
        ("bilstm", x_seq),
        ("cnn_lstm", x_cnn),
        ("attention_lstm", x_seq),
        ("transformer", x_seq),
    ]

    for name, x in test_cases:
        model = build_model(name, n_feat, n_cls).to(device)
        out = model(x)
        print(f"  {name}: input {tuple(x.shape)} → output {tuple(out.shape)}")

    print("\n✓ All models passed forward pass check.")
