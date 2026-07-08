# AI vs Human Music Classification

**Klasifikasi Musik Buatan Kecerdasan Artifisial menggunakan Convolutional Neural Network**

Classify a piece of music as **human-made** or **AI-generated** using a
Convolutional Neural Network **implemented from scratch in NumPy/CuPy** (no
`torch.nn` / Keras layers). The same CNN architecture is trained on **13 audio
feature representations** to compare which one best separates human from AI music.

A [Streamlit](https://streamlit.io/) app lets you explore the project, read each
model's evaluation, and run any of the 13 classifiers on your own audio — with a
visualisation of what the network "sees" at each convolutional layer.

---

## Highlights

- **CNN from scratch** — `im2col` convolutions, batch norm, adaptive pooling and
  hand-written backprop, all in NumPy (CuPy for GPU). See `cnn.py`.
- **13 feature front-ends** — Mel, STFT, Chroma, Bark spectrograms and
  MFCC / LFCC / PLP at 13, 20 and 40 coefficients. See `feature_extraction.py`.
- **Interactive UI** — upload audio, pick a feature, get a Human/AI verdict per
  5-second window, plus per-layer activation heatmaps. See `app/`.

## Results (test set, 3,224 clips)

| Feature | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| STFT | 0.9882 | 0.9882 | 0.9973 |
| Mel Spectrogram | 0.9857 | 0.9857 | 0.9950 |
| LFCC (40) | 0.9674 | 0.9674 | 0.9916 |
| LFCC (20) | 0.9178 | 0.9178 | 0.9645 |
| MFCC (40) | 0.9091 | 0.9091 | 0.9525 |
| LFCC (13) | 0.8933 | 0.8933 | 0.9565 |
| MFCC (20) | 0.8862 | 0.8861 | 0.9586 |
| Bark Spectrogram | 0.8772 | 0.8772 | 0.9440 |
| MFCC (13) | 0.8474 | 0.8473 | 0.9182 |
| PLP (20) | 0.8282 | 0.8280 | 0.9112 |
| PLP (40) | 0.8248 | 0.8247 | 0.9105 |
| PLP (13) | 0.8213 | 0.8212 | 0.9017 |
| Chromagram | 0.7444 | 0.7436 | 0.8210 |

Full classification reports and per-source error breakdowns are shown in the app
(and stored in `app/model_reports.json`).

## Project structure

```
.
├── cnn.py                  # From-scratch CNN: layers, training loop, evaluation
├── feature_extraction.py   # Audio loading, segmentation, 13 feature extractors
├── requirements.txt
├── ai_model/               # 13 trained models: best_<feature>_model.npz
├── images/                 # Per-model training history / confusion matrix / ROC
├── *_exp.ipynb             # One training-experiment notebook per feature
└── app/                    # Streamlit UI
    ├── app.py              # Page config + sidebar navigation
    ├── app_sections.py     # One render_* function per menu
    ├── app_content.py      # Static prose / tables (about, dataset, architecture)
    ├── inference_utils.py  # Model loading, feature extraction, prediction
    └── model_reports.json  # Test-set reports extracted from the notebooks
```

## Dataset

The models are trained on the secondary **FakeMusicCaps** dataset (from Kaggle),
all **16 kHz mono WAV**, licensed CC BY-NC 4.0:

- **Human** — `MusicCaps` (5,373 clips, sourced from AudioSet).
- **AI** — five open-source Text-To-Music generators, 5,521 clips each:
  `MusicGen`, `MusicLDM`, `AudioLDM2`, `Mustango`, `StableAudioOpen`.

The oversized AI side is down-sampled evenly across its five sources to balance the
classes, then split **70 : 15 : 15** (train / val / test). Each clip is cut into
non-overlapping **5-second** segments (zero-padded if short) and amplitude-normalised
before feature extraction.

## Installation

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU is optional: if [CuPy](https://cupy.dev/) is installed and a CUDA device is
available, `cnn.py` uses it automatically; otherwise it falls back to NumPy (CPU).

## Running the app

From the repository root:

```bash
streamlit run app/app.py
```

The sidebar has three menus:

1. **About This Project**
2. **Dataset and Model Architecture**
3. **Classifiers** — pick a feature, view its training results (charts +
   classification report + per-source breakdown), and run the classifier on an
   uploaded clip. After classifying, choose any 5-second window to see the model's
   per-layer activation heatmaps.

## Training from scratch

Training is driven by the notebooks (`<feature>_exp.ipynb`) and the pipeline in
`cnn.py`. You need the raw dataset laid out under `dataset/` as expected by
`feature_extraction.py`:

```
dataset/
├── music_caps/          # human (label 0)
├── music_gen_medium/    # AI (label 1)
├── music_ldm/
├── audio_ldm2/
├── mustango/
└── stable_audio_open/
```

Then, in a notebook:

```python
from feature_extraction import get_feature
from cnn import run_pipeline

data = get_feature("mel")              # extracts + caches features to mel_cache/
train_x, train_y, train_s = data["train"]
val_x,   val_y,   val_s   = data["val"]
test_x,  test_y,  test_s  = data["test"]

run_pipeline(train_x, train_y, val_x, val_y, test_x, test_y,
             labels=["Human", "AI"], scenario="mel",
             train_s=train_s, val_s=val_s, test_s=test_s)
```

This trains the CNN, saves the best checkpoint to `best_mel_model.npz`, and writes
the training-history / confusion-matrix / ROC-curve plots into `images/`.

## Model architecture

A single architecture is reused for all 13 features. Because the feature extractor
ends in an **adaptive average pool**, the network accepts any feature height.

**Feature extractor**

- `Conv2d(1→32, 3×3, pad 1)` → BatchNorm → ReLU → MaxPool 2×2
- `Conv2d(32→64, 3×3, pad 1)` → BatchNorm → ReLU → MaxPool 2×2
- `Conv2d(64→128, 3×3, pad 1)` → BatchNorm → ReLU → AdaptiveAvgPool 4×4

**Classifier head**

- Flatten (128 × 4 × 4 = 2048) → `Linear(2048→512)` → ReLU → Dropout 0.40
- `Linear(512→256)` → ReLU → Dropout 0.30 → `Linear(256→2)`

**Training** — Cross-Entropy (label smoothing 0.1), AdamW (lr 1e-3, wd 1e-4),
ReduceLROnPlateau, early stopping, batch 32, up to 50 epochs, seed 42.

## License

Dataset (FakeMusicCaps) is distributed under **CC BY-NC 4.0**. See the source
dataset for terms governing the audio.
