"""
Page renderers for the Streamlit UI. Each `render_*` draws one sidebar menu.
Layout only — inference lives in `inference_utils`, prose in `app_content`.
"""

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import app_content as content
import inference_utils as inference


# ══════════════════════════════════════════════════════════════════════════════
# Menu 1 — About This Project
# ══════════════════════════════════════════════════════════════════════════════

def render_about() -> None:
    st.title("🎵 Human vs AI Music Classifier")
    st.subheader(content.ABOUT_TITLE)
    st.markdown(content.ABOUT_MD)


# ══════════════════════════════════════════════════════════════════════════════
# Menu 2 — Dataset and Model Architecture
# ══════════════════════════════════════════════════════════════════════════════

def render_dataset_and_architecture() -> None:
    st.title("📚 Dataset and Model Architecture")

    # ── Dataset (Laporan Akhir §2.1) ──────────────────────────────────────────
    st.header("Dataset")
    st.caption("FakeMusicCaps · Laporan Akhir §2.1 (Akuisisi dan Eksplorasi Data)")
    st.markdown(content.DATASET_INTRO_MD)

    st.markdown("#### Table 1 — Dataset count & duration summary")
    table_rows = [
        dict(zip(content.DATASET_TABLE_COLUMNS, row))
        for row in content.DATASET_TABLE_ROWS
    ]
    st.dataframe(table_rows, width="stretch", hide_index=True)

    st.markdown(content.DATASET_NOTES_MD)

    st.divider()

    # ── Model description + architecture ──────────────────────────────────────
    st.header("Model Architecture")
    st.markdown(content.ARCHITECTURE_INTRO_MD)

    architecture_column, training_column = st.columns(2)
    with architecture_column:
        st.markdown("##### Feature extractor")
        for layer_line in content.ARCHITECTURE_FEATURE_LAYERS:
            st.markdown(f"- {layer_line}")
        st.markdown("##### Classifier head")
        for layer_line in content.ARCHITECTURE_CLASSIFIER_LAYERS:
            st.markdown(f"- {layer_line}")
    with training_column:
        st.markdown(content.TRAINING_MD)


# ══════════════════════════════════════════════════════════════════════════════
# Menu 3 — Classifiers
# ══════════════════════════════════════════════════════════════════════════════

def render_classifiers() -> None:
    st.title("🔍 Classifiers")

    available_keys = inference.available_feature_keys()
    if not available_keys:
        st.error(f"No trained models found in {inference.MODEL_DIRECTORY}.")
        return

    # ── Feature selector + description directly below the dropdown ─────────────
    selected_feature_key = st.selectbox(
        "Feature / Model",
        options=available_keys,
        format_func=lambda key: inference.FEATURE_CATALOGUE[key][0],
    )
    family = content.feature_family(selected_feature_key)
    st.markdown(content.FEATURE_DESCRIPTIONS[family])
    st.info(content.COMMON_FEATURE_NOTE)

    st.divider()

    results_tab, classifier_tab = st.tabs(["📊 Model Training Results", "🔍 Classifier"])
    with results_tab:
        _render_training_results(selected_feature_key)
    with classifier_tab:
        _render_classifier(selected_feature_key)


# ── Tab: Model Training Results ───────────────────────────────────────────────

def _render_training_results(feature_key: str) -> None:
    feature_label = inference.FEATURE_CATALOGUE[feature_key][0]
    st.markdown(f"Training report for **{feature_label}**.")

    images = dict(inference.report_image_paths(feature_key))

    # ── Charts ────────────────────────────────────────────────────────────────
    def _show_image(container, title: str) -> None:
        image_path = images.get(title)
        if image_path is not None and image_path.exists():
            container.image(str(image_path), caption=title, width="stretch")
        else:
            container.info(f"{title}\n\n(not found)")

    # Training history gets its own full-width row.
    _show_image(st, "Training history")

    # Confusion matrix + ROC curve share the next row.
    confusion_column, roc_column = st.columns(2)
    _show_image(confusion_column, "Confusion matrix")
    _show_image(roc_column, "ROC curve")

    # ── Evaluation on the test set ────────────────────────────────────────────
    st.divider()
    st.subheader("Evaluation on the test set")

    if inference.model_report(feature_key) is None:
        st.info("No saved evaluation report for this feature.")
        return

    metrics = inference.headline_metrics(feature_key)
    if metrics:
        metric_columns = st.columns(len(metrics))
        for metric_column, (metric_name, metric_value) in zip(metric_columns, metrics.items()):
            metric_column.metric(metric_name, metric_value)

    st.markdown("##### Classification report")
    st.dataframe(inference.classification_report_rows(feature_key),
                 width="stretch", hide_index=True)

    st.markdown("##### Per-source error breakdown")
    st.caption("→ Human / → AI = clips of that source misclassified into each label.")
    st.dataframe(inference.per_source_rows(feature_key),
                 width="stretch", hide_index=True)


# ── Tab: Classifier ───────────────────────────────────────────────────────────

def _render_classifier(feature_key: str) -> None:
    feature_label = inference.FEATURE_CATALOGUE[feature_key][0]

    st.markdown(content.HOW_IT_WORKS_MD)
    st.divider()

    st.markdown(
        f"Upload an audio clip; the **{feature_label}** model scores every "
        "5-second window and averages them into one verdict."
    )

    uploaded_audio = st.file_uploader(
        "Upload an audio clip",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Any length. Split into 5-second windows internally.",
        key=f"uploader_{feature_key}",
    )
    if uploaded_audio is None:
        return

    st.audio(uploaded_audio)

    # Identity of the current upload — changes whenever a different file is chosen,
    # so a cached result for a previous clip is not shown as if it were this one.
    audio_id = getattr(uploaded_audio, "file_id", None) \
        or f"{uploaded_audio.name}:{uploaded_audio.size}"

    # ── Classify on click; cache the result so the window selector below can
    #    re-render heatmaps without re-running the whole pipeline. ─────────────
    if st.button("🔍 Classify", type="primary", key=f"classify_{feature_key}"):
        file_suffix = Path(uploaded_audio.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_audio:
            temp_audio.write(uploaded_audio.getbuffer())
            temp_audio_path = temp_audio.name
        try:
            with st.spinner("Extracting features…"):
                input_tensor = inference.extract_windows(temp_audio_path, feature_key)
            with st.spinner("Running model…"):
                model = inference.load_model(feature_key)
                window_probabilities = inference.predict(model, input_tensor)  # (N, 2)
            st.session_state["classification_result"] = {
                "feature_key": feature_key,
                "audio_id": audio_id,
                "input_tensor": input_tensor,
                "window_probabilities": window_probabilities,
            }
        except Exception as inference_error:
            st.session_state.pop("classification_result", None)
            st.exception(inference_error)
        finally:
            Path(temp_audio_path).unlink(missing_ok=True)

    # Render the cached result only if it matches BOTH this feature and this clip.
    result = st.session_state.get("classification_result")
    if (not result
            or result["feature_key"] != feature_key
            or result.get("audio_id") != audio_id):
        return

    _render_prediction(result["window_probabilities"], result["input_tensor"].shape[0])
    _render_layer_activations(
        feature_key, result["input_tensor"], result["window_probabilities"]
    )


def _render_prediction(window_probabilities, window_count: int) -> None:
    ai_probabilities = window_probabilities[:, 1]
    mean_ai_probability = float(ai_probabilities.mean())
    verdict = "AI" if mean_ai_probability >= 0.5 else "Human"
    confidence = mean_ai_probability if verdict == "AI" else 1.0 - mean_ai_probability

    verdict_column, confidence_column, window_count_column = st.columns(3)
    verdict_column.metric("Verdict", verdict)
    confidence_column.metric("Confidence", f"{confidence * 100:.1f}%")
    window_count_column.metric("Windows analysed", window_count)

    st.progress(mean_ai_probability,
                text=f"Mean AI probability: {mean_ai_probability * 100:.1f}%")

    if verdict == "AI":
        st.error(f"🤖 Likely **AI-generated** ({confidence * 100:.1f}% confidence)")
    else:
        st.success(f"🎼 Likely **Human-made** ({confidence * 100:.1f}% confidence)")

    with st.expander("Per-window scores"):
        num_windows = len(ai_probabilities)
        per_window_table = {
            "Window": [
                f"{index * inference.WINDOW_SECONDS}"
                f"-{(index + 1) * inference.WINDOW_SECONDS}s"
                for index in range(num_windows)
            ],
            "P(Human)": window_probabilities[:, 0],
            "P(AI)":    window_probabilities[:, 1],
            "Label": [
                inference.CLASS_NAMES[int(probability >= 0.5)]
                for probability in ai_probabilities
            ],
        }
        st.dataframe(per_window_table, width="stretch", hide_index=True)
        st.bar_chart({"P(AI)": ai_probabilities})


# ── "What the model sees" — intermediate layer activations ────────────────────

def _heatmap_figure(matrix, title: str, figsize=(2.6, 2.6)):
    figure, axes = plt.subplots(figsize=figsize)
    axes.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
    axes.set_title(title, fontsize=8)
    axes.axis("off")
    figure.tight_layout(pad=0.2)
    return figure


def _render_layer_activations(feature_key, input_tensor, window_probabilities) -> None:
    st.divider()
    st.markdown("#### 🔬 What the model sees")

    num_windows = input_tensor.shape[0]
    ai_probabilities = window_probabilities[:, 1]

    def _window_label(index: int) -> str:
        start = index * inference.WINDOW_SECONDS
        end = (index + 1) * inference.WINDOW_SECONDS
        return f"Window {index + 1}  ({start}-{end}s)  ·  P(AI) = {ai_probabilities[index] * 100:.0f}%"

    if num_windows > 1:
        selected_window = st.selectbox(
            "Choose a 5-second window to visualise",
            options=list(range(num_windows)),
            format_func=_window_label,
            key=f"viz_window_{feature_key}",
        )
    else:
        selected_window = 0
        st.caption("The clip is a single 5-second window.")

    st.caption(
        "Activations for the selected window as it flows through the convolutional "
        "stack. Each panel is the mean over that layer's channels (frequency ↑, time →)."
    )

    model = inference.load_model(feature_key)
    activations = inference.layer_activations(
        model, input_tensor[selected_window:selected_window + 1]
    )

    activation_columns = st.columns(len(activations))
    for activation_column, (layer_label, activation) in zip(activation_columns, activations):
        mean_activation_map = activation.mean(axis=0)   # (C, H, W) -> (H, W)
        figure = _heatmap_figure(mean_activation_map, layer_label)
        activation_column.pyplot(figure)
        plt.close(figure)

    # Individual channels of the first conv block — the learned "filters" firing.
    _, block1_activation = activations[1]               # (32, H, W)
    channels_to_show = min(16, block1_activation.shape[0])
    with st.expander(f"Block 1 feature maps — first {channels_to_show} of "
                     f"{block1_activation.shape[0]} channels"):
        grid_figure, grid_axes = plt.subplots(4, 4, figsize=(6, 6))
        for channel_index, grid_axis in enumerate(grid_axes.flat):
            if channel_index < channels_to_show:
                grid_axis.imshow(block1_activation[channel_index], aspect="auto",
                                 origin="lower", cmap="magma")
            grid_axis.axis("off")
        grid_figure.tight_layout(pad=0.2)
        st.pyplot(grid_figure)
        plt.close(grid_figure)
