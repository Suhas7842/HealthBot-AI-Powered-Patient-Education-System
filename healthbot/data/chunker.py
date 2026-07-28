"""
Text chunking utilities for splitting medical documents.
Chunks text with overlap for better retrieval context.
"""

import re
from typing import List, Dict
from healthbot.config import settings


class TextChunker:
    """Chunks text into smaller segments for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap characters between chunks
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def split_on_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split on sentence boundaries (., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text into overlapping segments.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text) < self.chunk_size:
            # Return single chunk if text is short
            return [{
                "text": text,
                "chunk_id": 0,
                **(metadata or {})
            }]

        sentences = self.split_on_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    **(metadata or {})
                })

                # Start new chunk with overlap
                # Keep sentences that fit within overlap window
                overlap_text = ""
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_text = s + " " + overlap_text
                        overlap_length += len(s) + 1
                    else:
                        break

                current_chunk = [overlap_text.strip()] if overlap_text.strip() else []
                current_length = len(overlap_text)
                chunk_id += 1

            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                **(metadata or {})
            })

        return chunks

    def chunk_documents(self, documents: List[Dict], text_field: str = "abstract") -> List[Dict]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries
            text_field: Field name containing text to chunk

        Returns:
            List of chunked documents with metadata
        """
        all_chunks = []

        for doc in documents:
            text = doc.get(text_field, "")
            if not text:
                continue

            # Create metadata from document
            metadata = {
                k: v for k, v in doc.items()
                if k != text_field
            }

            # Chunk the text
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks


def main():
    """Test chunking functionality."""
    sample_text = """
    Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels.
    It occurs when the pancreas does not produce enough insulin or when the body cannot
    effectively use the insulin it produces. There are two main types of diabetes.
    Type 1 diabetes is an autoimmune condition where the immune system attacks
    insulin-producing cells. Type 2 diabetes is characterized by insulin resistance
    and is often associated with obesity and lifestyle factors. Common symptoms include
    increased thirst, frequent urination, fatigue, and blurred vision. Treatment may
    involve lifestyle modifications, oral medications, or insulin therapy depending on
    the type and severity of diabetes.
    """

    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_text(sample_text.strip(), {"source": "test"})

    print("="*80)
    print("TEXT CHUNKING DEMO")
    print("="*80)
    print(f"Original text length: {len(sample_text.strip())} characters")
    print(f"Number of chunks: {len(chunks)}")
    print("="*80)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i} ({len(chunk['text'])} chars):")
        print(f"{chunk['text'][:100]}...")

    print("="*80)


if __name__ == "__main__":
    main()
