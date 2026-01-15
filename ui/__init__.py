"""
UI components package.

Provides Streamlit-based interface components:
- Chat interface with streaming and citations
- Sidebar with system info and controls
- Document management interface
"""

from ui.chat import (
    clear_chat_button,
    display_chat_history,
    display_error_message,
    display_welcome_message,
    format_streaming_response,
    render_chat_input,
)
from ui.document_viewer import (
    render_document_list,
    render_document_manager,
    render_upload_interface,
)
from ui.sidebar import (
    render_document_stats,
    render_help_section,
    render_sidebar,
)

__all__ = [
    # Chat
    "display_chat_history",
    "render_chat_input",
    "display_welcome_message",
    "display_error_message",
    "format_streaming_response",
    "clear_chat_button",
    # Sidebar
    "render_sidebar",
    "render_document_stats",
    "render_help_section",
    # Document viewer
    "render_document_manager",
    "render_upload_interface",
    "render_document_list",
]