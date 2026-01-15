"""
Document management UI.

Provides interface for uploading, viewing, and deleting documents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import streamlit as st

from document_management import DocumentUploader

if TYPE_CHECKING:
    from document_management.indexer import DocumentIndexer
    from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = logging.getLogger(__name__)


def render_document_manager() -> None:
    """Render the complete document management interface."""
    st.header("📁 Document Management")

    tab1, tab2 = st.tabs(["📤 Upload", "📋 Manage"])

    with tab1:
        render_upload_interface()

    with tab2:
        render_document_list()


def render_upload_interface() -> None:
    """Render the document upload interface."""
    uploader = DocumentUploader()

    st.subheader("Upload Documents")

    st.info(
        f"📄 Allowed types: {', '.join(uploader.allowed_extensions)}\n\n"
        f"📦 Max size: {uploader.max_size_bytes // 1024 // 1024}MB"
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=uploader.allowed_extensions,
        accept_multiple_files=True,
        help="Upload one or more documents to index",
    )

    if uploaded_files:
        st.markdown(f"**Selected:** {len(uploaded_files)} file(s)")

        for file in uploaded_files:
            size_mb = file.size / 1024 / 1024
            st.text(f"• {file.name} ({size_mb:.2f}MB)")

        if st.button("📤 Upload & Index", type="primary", use_container_width=True):
            _process_uploads(uploaded_files, uploader)


def _process_uploads(
    uploaded_files: List[UploadedFile],
    uploader: DocumentUploader,
) -> None:
    """
    Process file uploads with progress feedback.

    Args:
        uploaded_files: List of uploaded file objects.
        uploader: DocumentUploader instance.
    """
    success_count = 0
    error_count = 0
    errors: List[str] = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")

        success, error, _ = uploader.save_file(file)

        if success:
            success_count += 1
            logger.info(f"Uploaded: {file.name}")
        else:
            error_count += 1
            errors.append(f"{file.name}: {error}")
            logger.warning(f"Upload failed for {file.name}: {error}")

        progress_bar.progress((i + 1) / len(uploaded_files))

    progress_bar.empty()
    status_text.empty()

    if success_count > 0:
        st.success(f"✅ Successfully uploaded {success_count} file(s)")
        st.info("🔄 Please restart the app to re-index documents")

    if error_count > 0:
        st.error(f"❌ Failed to upload {error_count} file(s)")
        with st.expander("Error Details"):
            for err in errors:
                st.text(err)


def render_document_list() -> None:
    """Render the list of existing documents."""
    uploader = DocumentUploader()
    documents = uploader.list_documents()

    st.subheader("Existing Documents")

    if not documents:
        st.info("No documents found. Upload some documents to get started!")
        return

    st.markdown(f"**Total:** {len(documents)} document(s)")

    total_size = sum(doc["size_mb"] for doc in documents)
    st.metric("Total Size", f"{total_size:.2f}MB")

    for doc in documents:
        _render_document_item(doc, uploader)


def _render_document_item(doc: Dict, uploader: DocumentUploader) -> None:
    """
    Render an individual document item with delete option.

    Args:
        doc: Document info dictionary.
        uploader: DocumentUploader instance.
    """
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            icon = _get_file_icon(doc["extension"])
            st.markdown(f"{icon} **{doc['name']}**")

        with col2:
            st.caption(f"{doc['size_mb']:.2f}MB • {doc['extension'].upper()}")

        with col3:
            if st.button("🗑️", key=f"delete_{doc['name']}", help="Delete document"):
                _delete_document(doc['name'], uploader)

        st.divider()


def _delete_document(filename: str, uploader: DocumentUploader) -> None:
    """
    Delete a document with confirmation.

    Args:
        filename: Name of file to delete.
        uploader: DocumentUploader instance.
    """
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = {}

    if filename not in st.session_state.confirm_delete:
        st.session_state.confirm_delete[filename] = False

    if not st.session_state.confirm_delete[filename]:
        st.warning(f"⚠️ Confirm deletion of '{filename}'?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ Confirm", key=f"confirm_yes_{filename}"):
                st.session_state.confirm_delete[filename] = True
                st.rerun()
        with col2:
            if st.button("✗ Cancel", key=f"confirm_no_{filename}"):
                st.session_state.confirm_delete.pop(filename, None)
                st.rerun()
    else:
        success, error = uploader.delete_file(filename)

        if success:
            st.success(f"✅ Deleted '{filename}'")
            st.info("🔄 Please restart the app to re-index")
            st.session_state.confirm_delete.pop(filename, None)
            logger.info(f"Deleted document: {filename}")
        else:
            st.error(f"❌ Failed to delete: {error}")
            st.session_state.confirm_delete.pop(filename, None)


def _get_file_icon(extension: str) -> str:
    """
    Get icon emoji for a file type.

    Args:
        extension: File extension (without dot).

    Returns:
        Emoji icon string.
    """
    icons = {
        'pdf': '📕',
        'txt': '📄',
        'docx': '📘',
        'md': '📝',
        'doc': '📘',
    }
    return icons.get(extension.lower(), '📄')


def render_indexing_stats(indexer: DocumentIndexer) -> None:
    """
    Display indexing statistics.

    Args:
        indexer: DocumentIndexer instance.
    """
    st.subheader("📊 Indexing Statistics")

    try:
        st.info("Indexing stats available after documents are indexed")
    except Exception as e:
        logger.error(f"Failed to display indexing stats: {e}")
        st.error("Unable to load indexing statistics")


def render_refresh_button() -> bool:
    """
    Render the index refresh button with confirmation.

    Returns:
        True if refresh was confirmed, False otherwise.
    """
    st.markdown("---")

    if st.button("🔄 Refresh Index", type="primary", use_container_width=True):
        st.warning("⚠️ This will rebuild the entire index")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ Confirm Refresh"):
                return True
        with col2:
            if st.button("✗ Cancel"):
                return False

    return False