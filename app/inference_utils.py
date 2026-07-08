"""
Inference core for the Streamlit UI.

Wraps the project's from-scratch NumPy CNN (`cnn.CNNModel`) and the training
feature pipeline (`feature_extraction.preprocess_clip`) so predictions in the
app match exactly how the models were trained.
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
# App lives in <root>/app/; models, images and cnn.py / feature_extraction.py
# live at the repo root, so put the root on sys.path before importing them.
APP_DIRECTORY = Path(__file__).parent
PROJECT_ROOT  = APP_DIRECTORY.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn import CNNModel, to_numpy, softmax              # noqa: E402  (needs sys.path)
import feature_extraction as feature_extraction_module  # noqa: E402

MODEL_DIRECTORY  = PROJECT_ROOT / "ai_model"
IMAGE_DIRECTORY  = PROJECT_ROOT / "images"
REPORTS_JSON     = APP_DIRECTORY / "model_reports.json"

# Test-set evaluation text (metrics + classification report + per-source
# breakdown) extracted once from the training notebooks.
MODEL_REPORTS = json.loads(REPORTS_JSON.read_text(encoding="utf-8")) \
    if REPORTS_JSON.exists() else {}

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES          = ["Human", "AI"]
WINDOW_SECONDS       = 5
INFERENCE_BATCH_SIZE = 32

# key == `scenario` slug used everywhere (model file, cache dir, image prefix).
# Each model file is  ai_model/best_<key>_model.npz
FEATURE_CATALOGUE = {
    "mel":     ("Mel Spectrogram",  "128-band log-mel spectrogram (dB)."),
    "stft":    ("STFT",             "Short-time Fourier power spectrogram (dB)."),
    "chroma":  ("Chromagram",       "12-bin pitch-class energy."),
    "bark":    ("Bark Spectrogram", "24-band Bark-scale spectrogram (dB)."),
    "mfcc":    ("MFCC (13)",        "13 Mel-frequency cepstral coefficients."),
    "mfcc_20": ("MFCC (20)",        "20 Mel-frequency cepstral coefficients."),
    "mfcc_40": ("MFCC (40)",        "40 Mel-frequency cepstral coefficients."),
    "lfcc":    ("LFCC (13)",        "13 Linear-frequency cepstral coefficients."),
    "lfcc_20": ("LFCC (20)",        "20 Linear-frequency cepstral coefficients."),
    "lfcc_40": ("LFCC (40)",        "40 Linear-frequency cepstral coefficients."),
    "plp":     ("PLP (13)",         "13 Perceptual Linear Prediction cepstra."),
    "plp_20":  ("PLP (20)",         "20 Perceptual Linear Prediction cepstra."),
    "plp_40":  ("PLP (40)",         "40 Perceptual Linear Prediction cepstra."),
}


def available_feature_keys() -> list[str]:
    """Feature keys whose trained `.npz` model is present on disk."""
    return [
        feature_key for feature_key in FEATURE_CATALOGUE
        if (MODEL_DIRECTORY / f"best_{feature_key}_model.npz").exists()
    ]


@st.cache_resource(show_spinner=False)
def load_model(feature_key: str) -> CNNModel:
    """Load a trained CNN for `feature_key`. `.load` wants the path without .npz."""
    model_path_without_extension = MODEL_DIRECTORY / f"best_{feature_key}_model"
    if not model_path_without_extension.with_suffix(".npz").exists():
        raise FileNotFoundError(f"Missing model: {model_path_without_extension}.npz")
    model = CNNModel(num_classes=2)
    model.load(str(model_path_without_extension))
    model.eval_mode()
    return model


def extract_windows(audio_file_path: str, feature_key: str) -> np.ndarray:
    """Audio file -> (num_windows, 1, height, width) float32 tensor.

    Mirrors the training pipeline: preprocess_clip already normalises,
    chunks into 5-s windows, computes the feature, and standardises each one.
    """
    feature_windows = feature_extraction_module.preprocess_clip(
        audio_file_path, feature_key
    )  # list of (height, width) arrays
    input_tensor = np.stack(feature_windows, axis=0).astype(np.float32)   # (N, H, W)
    input_tensor = input_tensor[..., np.newaxis]                          # (N, H, W, 1)
    input_tensor = input_tensor.transpose(0, 3, 1, 2).astype(np.float32)  # (N, 1, H, W)
    return input_tensor


def predict(model: CNNModel, input_tensor: np.ndarray) -> np.ndarray:
    """Return per-window class probabilities (num_windows, 2). Batched to bound memory."""
    probability_batches = []
    for batch_start in range(0, input_tensor.shape[0], INFERENCE_BATCH_SIZE):
        batch = input_tensor[batch_start:batch_start + INFERENCE_BATCH_SIZE]
        batch_probabilities = to_numpy(softmax(model.forward(batch)))
        probability_batches.append(batch_probabilities)
    return np.concatenate(probability_batches, axis=0)


def report_image_paths(feature_key: str) -> list[tuple[str, Path]]:
    """(title, path) for the three training-report images of a feature."""
    return [
        ("Training history", IMAGE_DIRECTORY / f"{feature_key}_training_history.png"),
        ("Confusion matrix", IMAGE_DIRECTORY / f"{feature_key}_confusion_matrix.png"),
        ("ROC curve",        IMAGE_DIRECTORY / f"{feature_key}_roc_curve.png"),
    ]


def model_report(feature_key: str) -> dict | None:
    """The structured test-set report (metrics, classification report, per-source)."""
    return MODEL_REPORTS.get(feature_key)


def _fmt(value, spec: str = ".4f") -> str:
    """Format a number for display; blank for a missing (None) value."""
    return "" if value is None else format(value, spec)


def headline_metrics(feature_key: str) -> dict[str, str]:
    """Top-line numbers for the metric cards."""
    metrics = MODEL_REPORTS.get(feature_key, {}).get("metrics", {})
    labels = {"Test Accuracy": "accuracy", "F1 Score": "f1", "ROC-AUC": "roc_auc"}
    return {label: _fmt(metrics[field]) for label, field in labels.items()
            if field in metrics}


def classification_report_rows(feature_key: str) -> list[dict[str, str]]:
    """Classification report as display-ready table rows."""
    rows = MODEL_REPORTS.get(feature_key, {}).get("classification_report", [])
    return [
        {"Class": row["class"], "Precision": _fmt(row["precision"]),
         "Recall": _fmt(row["recall"]), "F1-score": _fmt(row["f1"]),
         "Support": row["support"]}
        for row in rows
    ]


def per_source_rows(feature_key: str) -> list[dict[str, str]]:
    """Per-source error breakdown as display-ready table rows."""
    rows = MODEL_REPORTS.get(feature_key, {}).get("per_source", [])
    return [
        {"Source": row["source"], "Total": row["total"], "Wrong": row["wrong"],
         "Accuracy": f"{row['accuracy']:.2f}%",
         "→ Human": row["as_human"], "→ AI": row["as_ai"]}
        for row in rows
    ]


# Feature-extractor layers whose output is worth showing, keyed by their index
# in cnn.CNNModel.features (Conv → BN → ReLU → Pool blocks + final adaptive pool).
_ACTIVATION_CHECKPOINTS = {
    2:  "Block 1 · Conv+BN+ReLU (32 ch)",
    6:  "Block 2 · Conv+BN+ReLU (64 ch)",
    10: "Block 3 · Conv+BN+ReLU (128 ch)",
    11: "Adaptive AvgPool (128 × 4 × 4)",
}


def layer_activations(model: CNNModel, single_window: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Forward one window through the feature extractor, capturing intermediate
    activations so the UI can show "what the model sees".

    single_window: (1, 1, H, W).  Returns [(label, array of shape (C, H, W)), ...].
    """
    model.eval_mode()
    activations = [("Input feature", to_numpy(single_window)[0])]
    x = single_window
    for index, layer in enumerate(model.features.layers):
        x = layer.forward(x)
        if index in _ACTIVATION_CHECKPOINTS:
            activations.append((_ACTIVATION_CHECKPOINTS[index], to_numpy(x)[0]))
    return activations
