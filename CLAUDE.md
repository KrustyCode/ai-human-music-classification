# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

Human-vs-AI music classification. A CNN **written from scratch in NumPy/CuPy**
(no PyTorch/Keras layers) is trained on 13 audio feature representations; a
Streamlit app serves the trained models.

- `cnn.py` — the whole neural-net stack: layer classes (`Conv2d`, `BatchNorm2d`,
  `MaxPool2d`, `AdaptiveAvgPool2d`, `Linear`, …), `CNNModel`, `AdamW`,
  `CrossEntropyLoss`, training loop (`run_pipeline`, `train`, `evaluate`).
- `feature_extraction.py` — audio → feature tensors: loading (16 kHz mono),
  `chunk_into_windows` (5-s segments), 13 feature extractors, `preprocess_clip`,
  `get_feature` (extract + cache to `<feature>_cache/`).
- `ai_model/best_<feature>_model.npz` — 13 trained checkpoints.
- `images/` — per-model plots (`<feature>_{training_history,confusion_matrix,roc_curve}.png`).
- `*_exp.ipynb` — one training notebook per feature.
- `app/` — Streamlit UI (see below).

## Commands

```bash
pip install -r requirements.txt      # deps
streamlit run app/app.py             # launch the UI (from repo root)
```

There is no test suite. To sanity-check model I/O without the full deps, stub the
plotting/sklearn imports (`cnn.py` imports seaborn/matplotlib/sklearn at module
top but not in the forward/load path) and exercise `CNNModel.load` + `.forward`.

## Feature keys (the `scenario` slug)

One slug is used everywhere — model file, cache dir, image prefix, app catalogue:

```
mel  stft  chroma  bark
mfcc  mfcc_20  mfcc_40
lfcc  lfcc_20  lfcc_40
plp   plp_20   plp_40
```

Note the notebooks drop the underscore (`mfcc20_exp.ipynb`), but the slug uses it
(`mfcc_20`).

## Things that will bite you

- **Model load path has no extension.** `CNNModel.load(path)` calls
  `np.load(f"{path}.npz")`. Pass `ai_model/best_mel_model`, *not* `...model.npz`.
- **Classes: `0 = Human`, `1 = AI`.** `probs[:, 1]` is P(AI).
- **Adaptive pooling → any input height.** All 13 features share one architecture
  because `AdaptiveAvgPool2d((4,4))` normalises the spatial size. Don't hardcode H.
- **Inference must mirror training.** Audio → normalize → 5-s windows →
  feature → per-window standardise → `(N, 1, H, W)`. Reuse
  `feature_extraction.preprocess_clip`; do not reimplement.
- **Backend switch.** `cnn.py` imports CuPy as `xp` if available, else NumPy.
  Use `to_numpy(...)` before handing arrays to matplotlib/sklearn/Streamlit.

## App layout (`app/`)

- `app.py` — page config + sidebar tab-button nav (3 menus, selection in
  `st.session_state`).
- `app_sections.py` — one `render_*` per menu; all layout lives here.
- `app_content.py` — static prose/tables (about, dataset §2.1, architecture,
  per-feature descriptions).
- `inference_utils.py` — model loading (`@st.cache_resource`), feature extraction,
  prediction, report parsing, and `layer_activations` for the "what the model
  sees" heatmaps.
- `model_reports.json` — **structured** test-set reports per feature slug:
  `{"metrics": {...}, "classification_report": [...], "per_source": [...]}`.
  `inference_utils` reads it with plain dict access and only formats for display
  (no runtime text parsing).

Conventions for the app:

- The app lives in `app/` but `cnn.py` / `feature_extraction.py` / `ai_model/` /
  `images/` are at the **repo root**. `inference_utils.py` prepends the repo root
  to `sys.path`, and resolves `PROJECT_ROOT` as the parent of `app/`.
- Use the **new Streamlit width API**: `width="stretch"` / `width="content"`
  (not the deprecated `use_container_width=`).
- Descriptive, verbose variable names are preferred throughout the app modules.
- Cached classifier results in `st.session_state` are tagged with both the
  `feature_key` and an `audio_id` so a stale result is never shown after the
  feature or the uploaded clip changes.

## Regenerating `app/model_reports.json`

Source of truth is the notebook stdout: the `evaluate()` block containing
"Test Accuracy" + "Classification Report" + "Per-Source Error Breakdown". If a
model is retrained, parse that block into the structured shape above and store it
under the feature slug. Do the parsing **once, offline** (build step) — the app
reads structured JSON directly and does not parse text at runtime.
