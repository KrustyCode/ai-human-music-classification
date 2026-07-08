"""
Streamlit UI — Human vs AI Music Classifier
============================================

Sidebar-navigated app over the 13 from-scratch NumPy CNNs in `ai_model/`.

Menus:
    1. About This Project
    2. Dataset and Model Architecture
    3. Classifiers  (feature dropdown + description + tabs: results / classifier)

Run (from repo root):
    streamlit run app/app.py

Code layout (all under app/):
    app.py             — page config + sidebar navigation (this file)
    app_sections.py    — one render_* function per menu
    app_content.py     — static prose / tables (about, dataset, architecture, features)
    inference_utils.py — model loading + feature extraction + prediction
"""

import streamlit as st

import app_sections as sections

st.set_page_config(page_title="Human vs AI Music Classifier", page_icon="🎵",
                   layout="wide")

MENUS = {
    "About This Project":            sections.render_about,
    "Dataset and Model Architecture": sections.render_dataset_and_architecture,
    "Classifiers":                   sections.render_classifiers,
}

if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = next(iter(MENUS))


def _select_menu(menu_name: str) -> None:
    st.session_state.selected_menu = menu_name


with st.sidebar:
    st.header("🎵 Menu")
    for menu_name in MENUS:
        is_active = st.session_state.selected_menu == menu_name
        st.button(
            menu_name,
            key=f"nav_{menu_name}",
            width="stretch",
            type="primary" if is_active else "secondary",
            on_click=_select_menu,
            args=(menu_name,),
        )
    st.markdown("---")
    st.caption("Human vs AI Music Classification · CNN from scratch (NumPy/CuPy)")

MENUS[st.session_state.selected_menu]()
