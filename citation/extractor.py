"""Citation extraction and highlighting."""

import re
import logging
from typing import List, Dict
from config import settings

logger = logging.getLogger(__name__)


class CitationExtractor:
    """Extracts and highlights citations from retrieved context."""

    def __init__(self, highlight_length: int = None):
        """
        Initialize citation extractor.

        Args:
            highlight_length: Characters to show before/after citation
        """
        self.highlight_length = (
            highlight_length or settings.CITATION_HIGHLIGHT_LENGTH
        )

    def extract_citations(
        self,
        answer: str,
        source_nodes: List
    ) -> List[Dict]:
        """
        Extract citations from answer and link to sources.

        Args:
            answer: Generated answer text
            source_nodes: Retrieved source nodes

        Returns:
            List of citation dicts with source information
        """
        citations = []

        # Find quoted text in answer
        quoted_texts = re.findall(r'"([^"]*)"', answer)

        for node in source_nodes:
            source_text = node.node.text if hasattr(node, 'node') else node.text
            file_name = node.metadata.get('file_name', 'Unknown')
            page = node.metadata.get('page_label', 'N/A')
            score = getattr(node, 'score', None)

            # Check for exact matches
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
                        "score": score
                    })

            # Also add source even if not quoted
            if not any(c['source'] == file_name for c in citations):
                preview = source_text[:self.highlight_length * 2]
                citations.append({
                    "text": None,  # No direct quote
                    "source": file_name,
                    "page": page,
                    "highlight": preview,
                    "position": 0,
                    "score": score
                })

        logger.debug(f"Extracted {len(citations)} citations")
        return citations

    def format_citation_markdown(self, citation: Dict, index: int) -> str:
        """
        Format citation as markdown.

        Args:
            citation: Citation dict
            index: Citation number

        Returns:
            Formatted markdown string
        """
        score_str = f" (Score: {citation['score']:.3f})" if citation['score'] else ""

        if citation['text']:
            # Direct quote citation
            return f"""
**[{index}]** 📄 **{citation['source']}** (Page {citation['page']}){score_str}

> *"{citation['text']}"*

Context: ...{citation['highlight']}...
"""
        else:
            # General source citation
            return f"""
**[{index}]** 📄 **{citation['source']}** (Page {citation['page']}){score_str}

Preview: {citation['highlight']}...
"""