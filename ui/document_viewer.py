"""Document viewer UI for Streamlit.

Minimal viewer that shows file list and previews.
"""

import streamlit as st
from pathlib import Path
from config import settings


def render_document_viewer():
    data_dir = settings.DATA_DIR
    files = [p for p in data_dir.glob("**/*") if p.is_file()]

    st.header("📚 Documents")
    if not files:
        st.info("No documents found.")
        return

    for f in files:
        with st.expander(f.name):
            try:
                if f.suffix.lower() in [".txt", ".md"]:
                    st.code(f.read_text(encoding="utf-8", errors="ignore")[:1000])
                else:
                    st.write(f"File: {f.name} ({f.suffix})")
            except Exception as e:
                st.write(f"Unable to preview: {e}")
