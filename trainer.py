"""
Training Engine for Speech Emotion Recognition

Features:
- Mixed-precision training (AMP)
- Early stopping with patience
- Cosine annealing LR schedule with warm restarts
- Class-weighted cross-entropy for imbalanced data
- TensorBoard / CSV logging
- Checkpoint saving (best val accuracy and last epoch)
"""

import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from typing import Dict, List, Optional, Tuple
import csv
from collections import Counter


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class EmotionSpeechDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-extracted feature arrays.

    Args:
        X: Feature array of shape (N, n_features, T) for CNN, or (N, T, n_features) for LSTM
        y: Integer-encoded labels of shape (N,)
        augment: If True, apply data augmentation (time masking, noise injection)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time masking and Gaussian noise for regularization."""
        # Time masking: zero out a random 10% of time steps
        T = x.shape[-1]
        mask_len = max(1, int(T * 0.10))
        start = torch.randint(0, T - mask_len, (1,)).item()
        x = x.clone()
        if x.dim() == 2:
            x[:, start : start + mask_len] = 0.0
        else:
            x[start : start + mask_len] = 0.0

        # Additive Gaussian noise (σ = 0.02)
        noise = torch.randn_like(x) * 0.02
        return x + noise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        if self.augment and torch.rand(1).item() > 0.5:
            x = self._augment(x)
        return x, y


def build_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Create a weighted sampler to handle class imbalance."""
    class_counts = Counter(y)
    total = len(y)
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in y]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─── Training Utilities ───────────────────────────────────────────────────────

class EarlyStopping:
    """Monitor validation metric and stop training when no improvement."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._improved(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def _improved(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta


class MetricsLogger:
    """Log training metrics to CSV file."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.fieldnames = [
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr", "time_s"
        ]
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, metrics: Dict):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)


# ─── Trainer Class ────────────────────────────────────────────────────────────

class SERTrainer:
    """
    Complete training / evaluation pipeline for Speech Emotion Recognition.

    Args:
        model: nn.Module
        device: torch.device
        output_dir: Directory for checkpoints and logs
        class_names: List of emotion class names
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = "results",
        class_names: Optional[List[str]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir
        self.class_names = class_names or [str(i) for i in range(10)]
        os.makedirs(output_dir, exist_ok=True)

    def build_dataloaders(
        self,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size: int = 32,
        use_weighted_sampler: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Construct DataLoaders with optional weighted sampling."""
        train_ds = EmotionSpeechDataset(X_train, y_train, augment=True)
        val_ds = EmotionSpeechDataset(X_val, y_val, augment=False)
        test_ds = EmotionSpeechDataset(X_test, y_test, augment=False)

        sampler = build_weighted_sampler(y_train) if use_weighted_sampler else None

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        return train_loader, val_loader, test_loader

    def compute_class_weights(self, y_train: np.ndarray) -> torch.Tensor:
        """Compute inverse frequency class weights for loss."""
        counts = Counter(y_train)
        total = len(y_train)
        n_classes = len(self.class_names)
        weights = torch.tensor(
            [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
            dtype=torch.float32,
        ).to(self.device)
        return weights

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[optim.Optimizer],
        criterion: nn.Module,
        scaler: Optional[GradScaler] = None,
        train: bool = True,
    ) -> Tuple[float, float, float]:
        """Run a single epoch. Returns (loss, accuracy, f1)."""
        self.model.train(train)
        total_loss, all_preds, all_labels = 0.0, [], []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=scaler is not None):
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)

            if train:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return avg_loss, acc, f1

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        y_train: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        use_amp: bool = True,
        label_smoothing: float = 0.1,
    ) -> Dict:
        """
        Full training loop with AMP, early stopping, LR scheduling.

        Returns:
            Dictionary with training history and best metrics
        """
        # Loss with class weighting and label smoothing
        class_weights = self.compute_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        # AdamW optimizer
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

        scaler = GradScaler() if (use_amp and self.device.type == "cuda") else None
        early_stopper = EarlyStopping(patience=patience, mode="max")
        logger = MetricsLogger(os.path.join(self.output_dir, "training_log.csv"))

        best_val_acc = 0.0
        best_checkpoint = os.path.join(self.output_dir, "best_model.pth")
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

        print(f"\n{'='*60}")
        print(f"  Training: {epochs} epochs | LR: {lr} | Device: {self.device}")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc, _ = self._run_epoch(
                train_loader, optimizer, criterion, scaler, train=True
            )
            val_loss, val_acc, val_f1 = self._run_epoch(
                val_loader, None, criterion, scaler=None, train=False
            )
            scheduler.step()
            elapsed = time.time() - t0

            # Logging
            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_f1": round(val_f1, 4),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
                "time_s": round(elapsed, 2),
            }
            logger.log(metrics)

            for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]:
                history[k].append(metrics[k])

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "class_names": self.class_names,
                    },
                    best_checkpoint,
                )

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
                + (" ← best" if val_acc == best_val_acc else "")
            )

            if early_stopper(val_acc):
                print(f"\n[EarlyStopping] No improvement for {patience} epochs. Stopping.")
                break

        # Save last checkpoint
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "last_model.pth"))

        history["best_val_acc"] = best_val_acc
        with open(os.path.join(self.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n✓ Training complete. Best Val Acc: {best_val_acc:.4f}")
        return history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on the test set.
        Returns accuracy, F1, per-class report, and confusion matrix.
        """
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        report = classification_report(
            all_labels, all_preds, target_names=self.class_names, zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n{'='*60}")
        print(f"  TEST RESULTS")
        print(f"{'='*60}")
        print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  F1 Score : {f1:.4f}")
        print(f"\n{report}")

        results = {
            "accuracy": round(acc, 4),
            "weighted_f1": round(f1, 4),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        with open(os.path.join(self.output_dir, "test_results.json"), "w") as f:
            json.dump({k: v for k, v in results.items() if k != "confusion_matrix"}, f, indent=2)

        return results

    def load_best(self):
        """Load the best checkpoint saved during training."""
        ckpt = torch.load(
            os.path.join(self.output_dir, "best_model.pth"),
            map_location=self.device,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Checkpoint] Loaded best model (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
