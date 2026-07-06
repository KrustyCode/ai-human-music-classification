# System Documentation — Inference Layer & User Interface

This document describes two components of the Human-vs-AI music classification
system: the **inference layer** (`app/inference_utils.py`), which bridges the
trained models to the application, and the **user interface** (the `app/`
Streamlit package), which presents the models to the user.

---

## 1. Inference Layer (`inference_utils.py`)

### 1.1 Role

The inference layer is the single point of contact between the Streamlit UI and
the project's machine-learning core. It reuses the project's own modules —
`cnn.CNNModel` (the from-scratch NumPy/CuPy network) and
`feature_extraction.preprocess_clip` (the training feature pipeline) — so that
predictions produced in the app are computed in exactly the same way the models
were trained. No logic is re-implemented; the module only loads, orchestrates, and
formats for display. The network emits raw logits (`CNNModel.forward` has no final
softmax); the app obtains probabilities from the shared `cnn.softmax` — the same
numerically-stable function `cnn.py` uses in its loss and evaluation — so nothing
is duplicated.

### 1.2 Path resolution and import bridging

The application lives in `app/`, whereas the models (`ai_model/`), plots
(`images/`), and core modules (`cnn.py`, `feature_extraction.py`) live at the
repository root. On import, the module resolves `PROJECT_ROOT` as the parent of
`app/` and inserts it into `sys.path`, which allows `from cnn import ...` and
`import feature_extraction` to succeed regardless of the working directory. The
model, image, and report locations are all derived from `PROJECT_ROOT`.

### 1.3 Constants

| Constant | Meaning |
|---|---|
| `FEATURE_CATALOGUE` | Maps each feature slug (e.g. `mfcc_20`) to a `(display name, short description)` pair. The slug is the single identifier used for the model file, the cache directory, the image prefix, and the report key. |
| `CLASS_NAMES` | `["Human", "AI"]` — index 0 is Human, index 1 is AI. |
| `WINDOW_SECONDS` | 5 — the length of each analysis window. |
| `INFERENCE_BATCH_SIZE` | 32 — windows are scored in batches of this size to bound memory. |
| `MODEL_REPORTS` | The structured test-set evaluation report, loaded once from `model_reports.json`. |

### 1.4 Model discovery and loading

- **`available_feature_keys()`** returns only those feature slugs whose trained
  checkpoint (`ai_model/best_<slug>_model.npz`) is present on disk. The UI uses
  this to populate the feature selector, so the app never offers a model it
  cannot load.

- **`load_model(feature_key)`** instantiates a two-class `CNNModel`, loads the
  matching checkpoint, and puts the network in evaluation mode. It is decorated
  with `@st.cache_resource`, so each model is loaded from disk only once per
  session and reused across reruns. (Note: the checkpoint path is passed *without*
  the `.npz` extension, because `CNNModel.load` appends it internally.)

### 1.5 Inference functions

- **Probabilities** are produced by `cnn.softmax` (imported from `cnn.py`, not
  redefined here), which applies the numerically-stable max-shift softmax to the
  network's logits.

- **`extract_windows(audio_file_path, feature_key)`** turns an audio file into a
  model-ready tensor of shape `(num_windows, 1, height, width)`. It delegates to
  `preprocess_clip`, which loads the audio at 16 kHz mono, normalises amplitude,
  splits it into non-overlapping 5-second windows (zero-padding a short tail),
  computes the selected feature, and standardises each window. The result is then
  reshaped into the channel-first layout the network expects. Because this reuses
  the training pipeline verbatim, inference is guaranteed to mirror training.

- **`predict(model, input_tensor)`** runs the forward pass in batches and returns
  per-window class probabilities of shape `(num_windows, 2)`. Column 1 is the
  probability of the "AI" class.

### 1.6 Reporting functions

The training notebooks evaluated each model on the held-out test set and printed
the metrics, the classification report, and a per-source error breakdown. That
output was parsed **once, offline** into a structured `model_reports.json`, so the
application performs no text parsing at runtime — it reads plain JSON and formats
it for display.

- **`report_image_paths(feature_key)`** returns the `(title, path)` pairs for the
  three saved plots per model: training history, confusion matrix, and ROC curve.

- **`model_report(feature_key)`** returns the whole structured report for a
  feature (or `None` if absent).

- **`headline_metrics(feature_key)`** returns the three top-line figures —
  test accuracy, F1 score, and ROC-AUC — formatted for the metric cards.

- **`classification_report_rows(feature_key)`** returns the per-class /
  averaged classification report as display-ready table rows (Class, Precision,
  Recall, F1-score, Support).

- **`per_source_rows(feature_key)`** returns the per-source breakdown as table
  rows (Source, Total, Wrong, Accuracy, and the counts misclassified toward each
  label).

- **`_fmt(value, spec)`** is a small helper that formats numbers for display and
  renders a blank string for missing values (e.g. the accuracy row has no
  precision/recall).

### 1.7 Explainability (layer activations)

- **`_ACTIVATION_CHECKPOINTS`** names the feature-extractor layers whose output is
  worth showing: the three convolutional blocks (after Conv→BatchNorm→ReLU) and
  the final adaptive average pool.

- **`layer_activations(model, single_window)`** forwards a single window through
  the feature extractor layer by layer, capturing the output at each checkpoint.
  It returns a list of `(label, activation)` pairs, where each activation has
  shape `(channels, height, width)`. The first entry is the input feature itself.
  These arrays feed the "what the model sees" visualisation in the UI.

---

## 2. User Interface (the `app/` package)

### 2.1 Structure

The UI is deliberately split so that layout, content, and inference are
independent concerns:

| File | Responsibility |
|---|---|
| `app.py` | Page configuration and sidebar navigation. |
| `app_sections.py` | One `render_*` function per menu; all page layout lives here. |
| `app_content.py` | Static prose and tables (project blurb, dataset description, architecture, per-feature explanations). |
| `inference_utils.py` | Model loading, feature extraction, prediction, report access, and layer activations (documented above). |
| `model_reports.json` | Structured test-set reports keyed by feature slug. |

### 2.2 Navigation

`app.py` sets the page configuration and renders a sidebar that behaves like a set
of tab buttons. Each of the three menus is a full-width button; the currently
selected menu is styled as "primary" and the others as "secondary". Selection is
held in `st.session_state`, and each button uses an on-click callback so the
highlight and the page content update together on the same rerun. The chosen
menu's `render_*` function is then dispatched.

The three menus are: **About This Project**, **Dataset and Model Architecture**,
and **Classifiers**.

### 2.3 Menu 1 — About This Project

A static overview: the project title, its goal (build a human-vs-AI music
classifier and compare feature representations), and its motivation. Rendered
entirely from `app_content`.

### 2.4 Menu 2 — Dataset and Model Architecture

Two sections on one page:

- **Dataset** — a description of the FakeMusicCaps dataset (human vs. AI sources,
  sampling rate, licence), a summary table of class counts and clip durations, and
  notes on balancing, the 70/15/15 split, segmentation, and normalisation.

- **Model Architecture** — a description of the single CNN architecture shared by
  all feature models. The feature-extractor layers and the classifier-head layers
  are listed side by side with the training configuration (loss, optimiser,
  scheduler, early stopping, and so on).

### 2.5 Menu 3 — Classifiers

This page begins with a **feature selector** (a dropdown listing every feature
whose model is available). Immediately below the dropdown, the selected feature's
description is shown, so the choice and its explanation stay together. The page
then splits into two tabs.

#### 2.5.1 Tab — Model Training Results

Presents how the selected model performed during training and evaluation:

1. **Charts.** The training-history plot occupies its own full-width row; the
   confusion matrix and ROC curve share the row beneath it. Missing images degrade
   gracefully to an inline notice.
2. **Evaluation on the test set.** Below a divider: three metric cards
   (accuracy, F1, ROC-AUC), the **classification report** as a table, and the
   **per-source error breakdown** as a table. All values come from the structured
   report via the inference layer.

#### 2.5.2 Tab — Classifier

The interactive classifier. Flow:

1. A short "how it works" summary and an **audio uploader** (accepting common
   audio formats).
2. Once a file is uploaded, an audio player appears alongside a **Classify**
   button.
3. On click, the app writes the upload to a temporary file, extracts windows,
   loads the model, and computes per-window probabilities. The result — the input
   tensor and the probabilities — is stored in `st.session_state`, tagged with
   both the feature slug and an **audio identity** (the upload's file id, or a
   name/size fallback).
4. The cached result is rendered only if it matches **both** the current feature
   and the current clip. This ensures that changing the feature or uploading a
   different clip never displays a stale result; the previous output is hidden
   until the user classifies again.

The rendered result includes:

- **Verdict block** — the overall Human/AI verdict, a confidence percentage, and
  the number of windows analysed, followed by a progress bar showing the mean AI
  probability and a coloured success/error banner.
- **Per-window scores** — an expandable table of each window's probabilities and
  predicted label, plus a bar chart of per-window AI probability.

#### 2.5.3 "What the model sees" (layer activations)

Below the prediction, the Classifier tab visualises the network's internal
representations for the uploaded clip:

- When the clip has more than one window, a **window selector** lets the user
  choose which 5-second window to inspect; each option is labelled with its time
  range and that window's AI probability. Selecting a window triggers a rerun that
  regenerates the visualisation for that window only — an inexpensive single-window
  forward pass, made possible because the classification result is cached in
  session state.
- A row of **activation heatmaps** shows the selected window as it flows through
  the convolutional stack: the input feature, then each convolutional block, then
  the final adaptive pool. Each panel is the mean over that layer's channels
  (frequency on the vertical axis, time on the horizontal).
- An expandable grid shows the **individual channels of the first convolutional
  block** — a direct look at the learned filters responding to the input.

Heatmaps are drawn with Matplotlib (non-interactive "Agg" backend) and figures are
closed after rendering to avoid leaking memory.

### 2.6 State-management notes

- Models are cached with `@st.cache_resource`, so switching features or windows
  never reloads a model already in memory.
- Classification results are cached in `st.session_state` under a single key,
  tagged with the feature slug and the audio identity, which both prevents stale
  output and lets the window selector re-render activations without recomputing
  the whole pipeline.
- The UI uses Streamlit's current width API (`width="stretch"`) throughout, rather
  than the deprecated `use_container_width` argument.
