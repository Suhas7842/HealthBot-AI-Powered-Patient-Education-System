"""
Data processing pipeline for preparing medical documents.
Loads, cleans, and chunks documents for embedding and retrieval.
"""

from pathlib import Path

import pandas as pd

from healthbot.data.chunker import TextChunker
from healthbot.logger import logger


class DocumentProcessor:
    """Processes medical documents for RAG pipeline."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunker = TextChunker(chunk_size, chunk_overlap)

    def load_knowledge_base(
        self, path: str = "data/medical_kb.parquet"
    ) -> pd.DataFrame:
        """
        Load medical knowledge base from parquet file.

        Args:
            path: Path to knowledge base file

        Returns:
            DataFrame with medical articles
        """
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Knowledge base not found at {path}. "
                f"Run 'python -m healthbot.data.loader' first to build it."
            )

        df = pd.DataFrame(pd.read_parquet(path))
        logger.info(f"Loaded {len(df)} articles from {path}")
        return df

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and formatting.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove special characters but keep medical notation
        # Keep: letters, numbers, spaces, periods, commas, hyphens, parentheses
        text = "".join(c for c in text if c.isalnum() or c in " .,-()/%")

        return text.strip()

    def prepare_documents(self, df: pd.DataFrame) -> list[dict]:
        """
        Prepare documents for embedding by cleaning and structuring.

        Args:
            df: DataFrame with medical articles

        Returns:
            List of prepared document dictionaries
        """
        documents = []

        for _, row in df.iterrows():
            # Clean abstract text
            abstract = self.clean_text(row.get("abstract", ""))

            if not abstract or len(abstract) < 50:
                # Skip documents without meaningful abstracts
                continue

            doc = {
                "pmid": str(row.get("pmid", "")),
                "title": self.clean_text(row.get("title", "")),
                "abstract": abstract,
                "condition": row.get("condition", ""),
                "authors": row.get("authors", ""),
                "journal": row.get("journal", ""),
                "year": str(row.get("year", "")),
            }
            documents.append(doc)

        logger.info(f"Prepared {len(documents)} documents for processing")
        return documents

    def chunk_and_prepare(self, documents: list[dict]) -> list[dict]:
        """
        Chunk documents for embedding.

        Args:
            documents: List of document dictionaries

        Returns:
            List of chunks with metadata
        """
        chunks = self.chunker.chunk_documents(documents, text_field="abstract")
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def process_knowledge_base(
        self, kb_path: str = "data/medical_kb.parquet"
    ) -> list[dict]:
        """
        Complete processing pipeline: load, clean, and chunk knowledge base.

        Args:
            kb_path: Path to knowledge base parquet file

        Returns:
            List of processed and chunked documents ready for embedding
        """
        logger.info("Starting document processing pipeline")

        # Load knowledge base
        df = self.load_knowledge_base(kb_path)

        # Prepare documents
        documents = self.prepare_documents(df)

        # Chunk documents
        chunks = self.chunk_and_prepare(documents)

        logger.info(f"Processing complete: {len(chunks)} chunks ready for embedding")
        return chunks


def main():
    """Test document processing pipeline."""
    processor = DocumentProcessor()

    try:
        # Process knowledge base
        chunks = processor.process_knowledge_base()

        # Print summary
        print("\n" + "=" * 80)
        print("DOCUMENT PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total chunks: {len(chunks)}")
        print("\nSample chunk:")
        print(f"  PMID: {chunks[0]['pmid']}")
        print(f"  Condition: {chunks[0]['condition']}")
        print(f"  Text length: {len(chunks[0]['text'])} chars")
        print(f"  Text preview: {chunks[0]['text'][:200]}...")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the following command first:")
        print("  python -m healthbot.data.loader")


if __name__ == "__main__":
    main()
