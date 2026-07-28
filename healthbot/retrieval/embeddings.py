"""
Embedding manager using HuggingFace sentence transformers.
Handles text-to-vector conversion for semantic search.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from healthbot.config import settings
from healthbot.logger import logger


class EmbeddingManager:
    """Manages text embeddings using HuggingFace models."""

    def __init__(self, model_name: str = None):
        """
        Initialize embedding manager.

        Args:
            model_name: HuggingFace model name (defaults to config setting)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text")
            return [0.0] * self.get_embedding_dimension()

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()


def main():
    """Test embedding functionality."""
    manager = EmbeddingManager()

    # Test single embedding
    text = "Diabetes is a metabolic disorder characterized by high blood sugar."
    embedding = manager.embed_text(text)

    print("="*80)
    print("EMBEDDING MANAGER TEST")
    print("="*80)
    print(f"Model: {manager.model_name}")
    print(f"Embedding dimension: {manager.get_embedding_dimension()}")
    print(f"\nInput text: {text}")
    print(f"Embedding vector (first 10 values): {embedding[:10]}")
    print(f"Embedding shape: {len(embedding)}")

    # Test batch embedding
    texts = [
        "Hypertension is high blood pressure.",
        "Asthma affects the airways in the lungs.",
        "Heart disease refers to conditions affecting the heart."
    ]
    embeddings = manager.embed_batch(texts)
    print(f"\nBatch embedding: {len(embeddings)} texts embedded")
    print("="*80)


if __name__ == "__main__":
    main()
