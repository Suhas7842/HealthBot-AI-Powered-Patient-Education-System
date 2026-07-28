"""
Pinecone cloud vector store for production deployment.
Replaces local ChromaDB with scalable cloud-hosted vectors.
"""

import time

from pinecone import Pinecone, ServerlessSpec

from healthbot.config import settings
from healthbot.logger import logger
from healthbot.retrieval.embeddings import EmbeddingManager


class PineconeVectorStore:
    """Cloud-hosted vector store using Pinecone."""

    def __init__(
        self,
        index_name: str = "medical-knowledge",
        dimension: int = 384,
        metric: str = "cosine",
    ):
        """
        Initialize Pinecone vector store.

        Args:
            index_name: Name of the Pinecone index
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric

        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            logger.info("Pinecone client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager()

        # Get or create index
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")

            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",  # Free tier region
                ),
            )

            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status["ready"]:
                logger.info("Waiting for index to be ready...")
                time.sleep(1)

            logger.info(f"Index {self.index_name} created successfully")
        else:
            logger.info(f"Using existing index: {self.index_name}")

        # Connect to index
        self.index = self.pc.Index(self.index_name)

    def add_documents(self, documents: list[dict], batch_size: int = 100) -> None:
        """
        Add documents to Pinecone index.

        Args:
            documents: List of documents with 'text' and 'metadata'
            batch_size: Number of documents to upload per batch
        """
        logger.info(f"Adding {len(documents)} documents to Pinecone")

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Extract texts
            texts = [doc["text"] for doc in batch]

            # Generate embeddings
            embeddings = self.embedding_manager.embed_batch(texts)

            # Prepare vectors for Pinecone
            vectors = []
            for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                vector_id = f"doc_{i + j}"
                # Convert embedding to list if it's a numpy array
                emb_list = (
                    embedding.tolist() if hasattr(embedding, "tolist") else embedding
                )
                vectors.append(
                    {
                        "id": vector_id,
                        "values": emb_list,
                        "metadata": {
                            "text": doc["text"][:1000],  # Pinecone metadata limit
                            **doc.get("metadata", {}),
                        },
                    }
                )

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
            logger.info(f"Uploaded batch {i // batch_size + 1}: {len(vectors)} vectors")

        logger.info(f"Successfully added {len(documents)} documents to Pinecone")

    def similarity_search(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[dict]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query)

        # Convert to list if numpy array
        query_emb_list = (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else query_embedding
        )

        # Search Pinecone
        results = self.index.query(
            vector=query_emb_list, top_k=k, include_metadata=True, filter=filter
        )

        # Format results
        documents = []
        for match in results["matches"]:
            documents.append(
                {
                    "text": match["metadata"].get("text", ""),
                    "score": float(match["score"]),
                    "metadata": {
                        k: v
                        for k, v in match["metadata"].items()
                        if k != "text"  # Avoid duplicating text
                    },
                }
            )

        return documents

    def get_stats(self) -> dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats["total_vector_count"],
            "dimension": stats["dimension"],
            "index_fullness": stats.get("index_fullness", 0.0),
        }


def migrate_chromadb_to_pinecone():
    """
    Migrate existing ChromaDB data to Pinecone.
    Run this once to upload your 2,578 chunks.
    """
    from healthbot.retrieval.vector_store import MedicalVectorStore

    logger.info("Starting ChromaDB → Pinecone migration")

    # Load from ChromaDB
    chroma_store = MedicalVectorStore()
    collection_data = chroma_store.collection.get()

    # Prepare documents
    documents = []
    for i, (text, metadata) in enumerate(
        zip(collection_data["documents"], collection_data["metadatas"])
    ):
        documents.append({"text": text, "metadata": metadata or {}})

    logger.info(f"Loaded {len(documents)} documents from ChromaDB")

    # Upload to Pinecone
    pinecone_store = PineconeVectorStore()
    pinecone_store.add_documents(documents)

    # Verify
    stats = pinecone_store.get_stats()
    logger.info(f"Migration complete! Pinecone stats: {stats}")


if __name__ == "__main__":
    # Test or migrate
    migrate_chromadb_to_pinecone()
