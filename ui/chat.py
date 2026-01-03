"""Simple chat UI helpers for Streamlit.

These are minimal wrappers to keep `app.py` organized.
"""

import streamlit as st
from typing import List, Dict


def render_chat(messages: List[Dict]):
    """Render a list of chat messages (role/content dicts)."""
    for message in messages:
        with st.chat_message(message.get("role", "user")):
            st.markdown(message.get("content", ""))


def stream_response(generator):
    """Stream response generator to Streamlit write.

    Accepts an iterable/generator of strings and returns the concatenated text.
    """
    full = ""
    for chunk in generator:
        st.write(chunk)
        full += chunk
    return full
