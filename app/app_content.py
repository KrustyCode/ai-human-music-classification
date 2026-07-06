"""
Static, human-readable content for the Streamlit UI: project blurb, dataset
description (Laporan Akhir §2.1, pp. 10-11), model architecture, and per-feature
explanations. Kept apart from layout so the section modules stay short.
"""

# ══════════════════════════════════════════════════════════════════════════════
# About This Project
# ══════════════════════════════════════════════════════════════════════════════

ABOUT_TITLE = "Klasifikasi Musik Buatan Kecerdasan Artifisial menggunakan CNN"

ABOUT_MD = """
This project builds an automatic classifier that distinguishes **human-made
music** from **AI-generated music** using a **Convolutional Neural Network (CNN)**
implemented **from scratch in NumPy/CuPy** — no `torch.nn` / Keras layers.

The core research question: *which audio feature representation lets a CNN best
separate human from AI music?* To answer it, the same CNN architecture is trained
on **13 feature front-ends** (Mel, STFT, Bark, Chroma, and MFCC / LFCC / PLP at
several coefficient counts) and their evaluation metrics are compared.

**Goals**
1. Build an automatic human-vs-AI music classifier using CNNs.
2. Compare per-feature evaluation results to find the best-performing
   representation.

**Why it matters**
- Adds to research on AI-music detection with deep learning.
- Supports building automatic detectors that tell human and AI music apart.
- Shows how different audio-feature representations affect model performance.

Use the sidebar to read about the **Dataset**, the **Model & Classifiers**
(architecture + a live classifier you can try on your own audio).
"""

# ══════════════════════════════════════════════════════════════════════════════
# Dataset  (Laporan Akhir §2.1 Akuisisi dan Eksplorasi Data, pp. 10-11)
# ══════════════════════════════════════════════════════════════════════════════

DATASET_INTRO_MD = """
The study uses the secondary **FakeMusicCaps** dataset (from Kaggle). It contains
audio files grouped by source and type — **human-made** vs **AI-generated** music.
All audio is **WAV**, uniformly sampled at **16,000 Hz**, distributed under the
**CC BY-NC 4.0** licence.

- **AI music** — five open-source Text-To-Music (TTM) generators, each with
  **5,521** clips: **MusicGen, MusicLDM, AudioLDM2, Mustango, StableAudioOpen**.
  (A sixth class, *SunoCaps* — 63 clips from the commercial Suno AI — was dropped
  for being far too small.)
- **Human music** — the **MusicCaps** class: **5,373** clips sourced from
  **AudioSet**.
"""

# Tabel 1 — Ringkasan Jumlah dan Durasi Dataset (durations in seconds)
DATASET_TABLE_COLUMNS = ["Class", "Count", "Min (s)", "Median (s)", "Mean (s)", "Max (s)", "Std (s)"]
DATASET_TABLE_ROWS = [
    ["MusicCaps (Human)", "5,373", "1.53", "10.01", "10.00", "10.01", "0.16"],
    ["MusicGen",          "5,521", "10.18", "10.18", "10.18", "10.18", "0.00"],
    ["MusicLDM",          "5,521", "10.00", "10.00", "10.00", "10.00", "0.00"],
    ["AudioLDM2",         "5,521", "10.00", "10.00", "10.00", "10.00", "0.00"],
    ["Mustango",          "5,521", "10.24", "10.24", "10.24", "10.24", "0.00"],
    ["StableAudioOpen",   "5,521", "10.00", "10.00", "10.00", "10.00", "0.00"],
]

DATASET_NOTES_MD = """
**Duration.** Most clips are ~10 s. The five AI classes are near-constant length
(std 0.00 s); MusicCaps varies more (1.53–10.01 s, std 0.16 s) — a few human clips
are shorter than the rest.

**Sample rate.** Every file is **16 kHz**, so temporal and spectral characteristics
are comparable across classes without resolution differences skewing the results.

**Balancing.** The raw AI side totals **27,605** clips — far more than the human
side. To avoid class imbalance, the AI class is **down-sampled evenly across its
five sources** to match the human count.

**Split.** After balancing, data is split **70 : 15 : 15** into train / validation
/ test.

**Segmentation.** Each file is cut into non-overlapping **5-second** segments; a
final segment shorter than 5 s is **zero-padded**. This equalises input length and
increases the number of training samples.

**Normalisation.** Waveform amplitude is normalised before feature extraction so
the model focuses on class-relevant characteristics rather than loudness.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Model Description + Architecture
# ══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE_INTRO_MD = """
A single CNN architecture (built from scratch in NumPy/CuPy — im2col convolutions
and hand-written backprop) is trained separately for each of the 13 audio features.
Every feature produces a single-channel 2-D "image" `(1, H, W)`; because the
feature extractor ends in an **adaptive average pool**, the network accepts any
feature height `H`, so all 13 features share the exact same layer stack.

Input clips are **binary-classified**: `0 = Human`, `1 = AI`.
"""

# Layer stack mirrors cnn.CNNModel (features + classifier).
ARCHITECTURE_FEATURE_LAYERS = [
    "Conv2d(1 → 32, 3×3, pad 1)  →  BatchNorm  →  ReLU  →  MaxPool 2×2",
    "Conv2d(32 → 64, 3×3, pad 1)  →  BatchNorm  →  ReLU  →  MaxPool 2×2",
    "Conv2d(64 → 128, 3×3, pad 1)  →  BatchNorm  →  ReLU  →  AdaptiveAvgPool 4×4",
]
ARCHITECTURE_CLASSIFIER_LAYERS = [
    "Flatten (128 × 4 × 4 = 2048)",
    "Linear(2048 → 512)  →  ReLU  →  Dropout 0.40",
    "Linear(512 → 256)  →  ReLU  →  Dropout 0.30",
    "Linear(256 → 2)   (Human / AI logits)",
]

TRAINING_MD = """
**Training setup**
- Loss: Cross-Entropy with **label smoothing 0.1**
- Optimiser: **AdamW** (lr 1e-3, weight decay 1e-4)
- Scheduler: **ReduceLROnPlateau** (factor 0.5, patience 5, on val loss)
- **Early stopping** (patience 10) on validation loss; best checkpoint saved
- Batch size 32, up to 50 epochs, fixed seed 42
- Backend: **CuPy (GPU)** if available, else **NumPy (CPU)**

**Inference in this app.** Uploaded audio → 16 kHz mono → 5-s windows → selected
feature → per-window standardisation → CNN → softmax. The per-window `P(AI)` scores
are **averaged** into one verdict.
"""

HOW_IT_WORKS_MD = """
**How it works**
1. Audio → 16 kHz mono
2. Split into 5-second windows
3. Compute the selected feature
4. CNN scores each window
5. Average the per-window scores → final verdict
"""

# ══════════════════════════════════════════════════════════════════════════════
# Feature descriptions  (keyed by family; variants share a description)
# ══════════════════════════════════════════════════════════════════════════════

COMMON_FEATURE_NOTE = (
    "All features are computed on 16 kHz mono, 5-second windows, then "
    "**standardised** (zero-mean, unit-variance) per window before the CNN."
)

FEATURE_DESCRIPTIONS = {
    "mel": """
### Mel Spectrogram
A **128-band log-mel spectrogram** in dB. The STFT power spectrum
(`n_fft=1024`, `hop=256`, Hann window) is warped onto the perceptual **Mel**
frequency scale, then converted to decibels. Captures how energy is distributed
across perceptually spaced frequency bands over time — a strong general-purpose
representation for music.
""",
    "stft": """
### STFT (Short-Time Fourier Transform)
The raw time–frequency **power spectrogram** in dB (`n_fft=512`, `hop=256`),
giving 257 linear frequency bins. The least "processed" feature here: it keeps
fine spectral detail that learned filters can exploit, at the cost of higher
dimensionality.
""",
    "chroma": """
### Chromagram
**12-bin pitch-class energy** (`chroma_stft`, `n_fft=1024`, `hop=256`). Folds all
octaves onto the 12 semitones (C, C#, …, B), summarising **harmonic / tonal**
content while discarding timbre. Compact but harmony-focused.
""",
    "bark": """
### Bark Spectrogram
A **24-band Bark-scale spectrogram** in dB. STFT power (`n_fft=1024`, `hop=256`)
is folded through a rectangular **Bark** filterbank — a psychoacoustic critical-band
scale — then converted to dB. Similar idea to Mel but on the Bark scale.
""",
    "mfcc": """
### MFCC (Mel-Frequency Cepstral Coefficients)
The **DCT** of the log-mel spectrum, keeping the first *N* cepstral coefficients
(`n_fft=1024`, `hop=256`). Classic, compact **timbre** descriptor. Three variants
are trained: **13, 20, 40** coefficients — more coefficients retain finer spectral
envelope detail.
""",
    "lfcc": """
### LFCC (Linear-Frequency Cepstral Coefficients)
Like MFCC but using a **linearly-spaced** (128-filter) filterbank instead of Mel,
followed by a DCT-II (`norm='ortho'`), keeping the first *N* coefficients. Gives
even weight to high frequencies. Variants: **13, 20, 40** coefficients.
""",
    "plp": """
### PLP (Perceptual Linear Prediction)
A perceptually-motivated cepstrum: Bark-warped power spectrum → **equal-loudness
weighting** → **cube-root** intensity compression → **LPC** (Levinson–Durbin,
order 12) → cepstral coefficients. Variants: **13, 20, 40** coefficients.
""",
}


def feature_family(feature_key: str) -> str:
    """'mfcc_20' -> 'mfcc'.  Base family used to look up a description."""
    return feature_key.split("_")[0]
