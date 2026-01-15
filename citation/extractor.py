"""
Citation extraction and highlighting.

Extracts quoted text from LLM responses and links them back to source documents.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


class CitationExtractor:
    """
    Extracts and formats citations from generated answers.

    Identifies quoted text in responses and links them to source documents,
    providing context for verification.

    Attributes:
        highlight_length: Characters to show before/after matched text.
    """

    __slots__ = ('highlight_length',)

    def __init__(self, highlight_length: Optional[int] = None) -> None:
        """
        Initialize the citation extractor.

        Args:
            highlight_length: Context characters around citations.
                Defaults to settings.CITATION_HIGHLIGHT_LENGTH.
        """
        self.highlight_length: int = (
            highlight_length or settings.CITATION_HIGHLIGHT_LENGTH
        )

    def extract_citations(
        self,
        answer: str,
        source_nodes: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract citations from an answer and link to sources.

        Args:
            answer: Generated answer text.
            source_nodes: Retrieved source nodes (NodeWithScore objects).

        Returns:
            List of citation dictionaries with source information.
        """
        citations: List[Dict[str, Any]] = []

        # Find quoted text in the answer
        quoted_texts = re.findall(r'"([^"]*)"', answer)

        for node in source_nodes:
            source_text = node.node.text if hasattr(node, 'node') else node.text
            metadata = node.node.metadata if hasattr(node, 'node') else getattr(node, 'metadata', {})
            file_name = metadata.get('file_name', 'Unknown')
            page = metadata.get('page_label', 'N/A')
            score = getattr(node, 'score', None)

            # Check for quoted text matches
            for quote in quoted_texts:
                if len(quote) < 10:  # Skip very short quotes
                    continue

                if quote.lower() in source_text.lower():
                    pos = source_text.lower().index(quote.lower())
                    start = max(0, pos - self.highlight_length)
                    end = min(len(source_text), pos + len(quote) + self.highlight_length)

                    citations.append({
                        "text": quote,
                        "source": file_name,
                        "page": page,
                        "highlight": source_text[start:end],
                        "position": pos,
                        "score": score,
                    })

            # Add source even if not quoted (for reference)
            if not any(c['source'] == file_name for c in citations):
                preview = source_text[:self.highlight_length * 2]
                citations.append({
                    "text": None,
                    "source": file_name,
                    "page": page,
                    "highlight": preview,
                    "position": 0,
                    "score": score,
                })

        logger.debug(f"Extracted {len(citations)} citations")
        return citations

    def format_citation_markdown(self, citation: Dict[str, Any], index: int) -> str:
        """
        Format a citation as markdown.

        Args:
            citation: Citation dictionary.
            index: Citation number for display.

        Returns:
            Formatted markdown string.
        """
        score_str = f" (Score: {citation['score']:.3f})" if citation['score'] else ""

        if citation['text']:
            return f"""
**[{index}]** 📄 **{citation['source']}** (Page {citation['page']}){score_str}

> *"{citation['text']}"*

Context: ...{citation['highlight']}...
"""
        else:
            return f"""
**[{index}]** 📄 **{citation['source']}** (Page {citation['page']}){score_str}

Preview: {citation['highlight']}...
"""