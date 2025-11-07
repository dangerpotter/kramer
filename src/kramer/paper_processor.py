"""Paper processor for downloading and extracting text from PDFs."""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict
import re

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

import httpx


class PaperProcessor:
    """
    Handles downloading PDFs and extracting text from scientific papers.

    Uses Semantic Scholar API for PDF downloads and PyMuPDF for text extraction.
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize the paper processor.

        Args:
            temp_dir: Directory to store temporary PDF files.
                     If None, uses system temp directory.
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "kramer_papers"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def download_pdf(
        self,
        paper_id: str,
        s2_api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Download a PDF from Semantic Scholar.

        Args:
            paper_id: Semantic Scholar paper ID
            s2_api_key: Optional API key for Semantic Scholar

        Returns:
            Path to downloaded PDF file, or None if unavailable
        """
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/pdf"

        headers = {}
        if s2_api_key:
            headers["x-api-key"] = s2_api_key

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)

                # Check if PDF is available
                if response.status_code == 404:
                    return None

                response.raise_for_status()

                # Save PDF to temp directory
                pdf_path = self.temp_dir / f"{paper_id}.pdf"
                pdf_path.write_bytes(response.content)

                return str(pdf_path)

        except httpx.HTTPError as e:
            print(f"Error downloading PDF for {paper_id}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error downloading PDF for {paper_id}: {e}")
            return None

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text from all pages

        Raises:
            ImportError: If PyMuPDF is not installed
            FileNotFoundError: If PDF file doesn't exist
        """
        if fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is not installed. "
                "Install it with: pip install pymupdf"
            )

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            text_parts = []

            for page_num, page in enumerate(doc):
                # Extract text from page
                page_text = page.get_text()

                # Clean the text
                cleaned_text = self._clean_text(page_text, page_num)

                if cleaned_text.strip():
                    text_parts.append(cleaned_text)

            doc.close()

            return "\n\n".join(text_parts)

        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {e}")

    def _clean_text(self, text: str, page_num: int) -> str:
        """
        Clean extracted text by removing headers, footers, and extra whitespace.

        Args:
            text: Raw text from PDF page
            page_num: Page number (0-indexed)

        Returns:
            Cleaned text
        """
        # Split into lines
        lines = text.split("\n")

        # Remove common header/footer patterns
        cleaned_lines = []
        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip lines that are just page numbers
            if re.match(r"^\d+$", line):
                continue

            # Skip very short lines at top/bottom (likely headers/footers)
            if (i < 2 or i >= len(lines) - 2) and len(line) < 10:
                continue

            cleaned_lines.append(line)

        # Join with spaces and normalize whitespace
        text = " ".join(cleaned_lines)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks for better retrieval.

        Args:
            text: Full text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks

        Returns:
            List of chunk dictionaries with:
                - chunk_id: Unique identifier for the chunk
                - text: The chunk text
                - start_char: Starting character position in original text
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # Get chunk with overlap
            end = start + chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary if not at end
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                last_period = chunk_text.rfind(". ")
                last_question = chunk_text.rfind("? ")
                last_exclamation = chunk_text.rfind("! ")

                last_sentence_end = max(last_period, last_question, last_exclamation)

                # If we found a sentence ending and it's not too early, break there
                if last_sentence_end > chunk_size * 0.7:
                    chunk_text = chunk_text[:last_sentence_end + 2]
                    end = start + last_sentence_end + 2

            # Create chunk entry
            chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "text": chunk_text.strip(),
                "start_char": start
            })

            # Move to next chunk with overlap
            start = end - overlap
            chunk_id += 1

            # Prevent infinite loop
            if start >= len(text):
                break

        return chunks

    def cleanup(self, paper_id: Optional[str] = None):
        """
        Clean up temporary PDF files.

        Args:
            paper_id: If provided, only delete PDF for this paper.
                     If None, delete all PDFs in temp directory.
        """
        if paper_id:
            pdf_path = self.temp_dir / f"{paper_id}.pdf"
            if pdf_path.exists():
                pdf_path.unlink()
        else:
            # Delete all PDFs in temp directory
            for pdf_file in self.temp_dir.glob("*.pdf"):
                pdf_file.unlink()
