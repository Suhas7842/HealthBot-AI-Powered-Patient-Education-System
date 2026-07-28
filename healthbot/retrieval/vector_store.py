"""
ChromaDB vector store for semantic search over medical documents.
Persists embeddings locally for efficient retrieval.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
from healthbot.config import settings
from healthbot.retrieval.embeddings import EmbeddingManager
from healthbot.data.processor import DocumentProcessor
from healthbot.logger import logger


class MedicalVectorStore:
    """ChromaDB-based vector store for medical document retrieval."""

    def __init__(self, collection_name: str = "medical_knowledge"):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.embedding_manager = EmbeddingManager()

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Medical articles from PubMed"}
        )

        logger.info(f"Vector store initialized: {self.collection.count()} documents")

    def add_documents(
        self,
        documents: List[Dict],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document dictionaries with 'text' field
            batch_size: Number of documents to process per batch
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Extract texts and metadata
            texts = [doc["text"] for doc in batch]
            metadatas = []
            ids = []

            for j, doc in enumerate(batch):
                # Create unique ID
                doc_id = f"{doc.get('pmid', 'unknown')}_{doc.get('chunk_id', j)}"
                ids.append(doc_id)

                # Prepare metadata (ChromaDB requires string values)
                metadata = {
                    "pmid": str(doc.get("pmid", "")),
                    "title": str(doc.get("title", ""))[:500],  # Limit length
                    "condition": str(doc.get("condition", "")),
                    "chunk_id": str(doc.get("chunk_id", "")),
                    "year": str(doc.get("year", "")),
                }
                metadatas.append(metadata)

            # Generate embeddings
            embeddings = self.embedding_manager.embed_batch(texts)

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            logger.info(f"Batch {i//batch_size + 1}: Added {len(batch)} documents")

        logger.info(f"Total documents in store: {self.collection.count()}")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        condition_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query: Search query text
            k: Number of results to return
            condition_filter: Optional condition to filter results

        Returns:
            List of dictionaries with document text, metadata, and scores
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query)

        # Build filter if condition specified
        where_filter = None
        if condition_filter:
            where_filter = {"condition": condition_filter}

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter
        )

        # Format results
        documents = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                doc = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                    "pmid": results["metadatas"][0][i].get("pmid", ""),
                    "title": results["metadatas"][0][i].get("title", ""),
                    "condition": results["metadatas"][0][i].get("condition", ""),
                }
                documents.append(doc)

        logger.info(f"Semantic search returned {len(documents)} results")
        return documents

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Medical articles from PubMed"}
        )
        logger.info("Vector store reset")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_manager.get_embedding_dimension()
        }


def build_vector_store():
    """Build vector store from processed knowledge base."""
    logger.info("Building vector store from knowledge base")

    # Process documents
    processor = DocumentProcessor()
    chunks = processor.process_knowledge_base()

    # Initialize vector store
    store = MedicalVectorStore()

    # Reset if rebuilding
    if store.collection.count() > 0:
        logger.info("Existing collection found. Resetting...")
        store.reset()

    # Add documents
    store.add_documents(chunks)

    # Print stats
    stats = store.get_stats()
    print("\n" + "="*80)
    print("VECTOR STORE BUILD COMPLETE")
    print("="*80)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Collection: {stats['collection_name']}")
    print("="*80)


def main():
    """Main function to build or test vector store."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_vector_store()
    else:
        # Test search
        store = MedicalVectorStore()

        if store.collection.count() == 0:
            print("\nVector store is empty. Build it first:")
            print("  python -m healthbot.retrieval.vector_store build")
            return

        # Test query
        query = "What are the symptoms of diabetes?"
        print(f"\nQuerying: '{query}'")
        print("="*80)

        results = store.similarity_search(query, k=3)

        for i, doc in enumerate(results, 1):
            print(f"\nResult {i} (score: {doc['score']:.3f}):")
            print(f"  Condition: {doc['condition']}")
            print(f"  Title: {doc['title'][:80]}...")
            print(f"  Text: {doc['text'][:200]}...")

        print("="*80)


if __name__ == "__main__":
    main()
