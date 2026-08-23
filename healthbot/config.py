"""
Configuration management using Pydantic settings.
Loads environment variables from .env or config.env files.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration (OpenAI-compatible or Gemini)
    LLM_PROVIDER: str = "openai"  # "openai", "groq", or "gemini"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0
    OPENAI_MAX_TOKENS: int = 1000
    OPENAI_BASE_URL: str | None = None  # For Groq: https://api.groq.com/openai/v1

    # Google Gemini Configuration
    GOOGLE_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-3.6-flash"  # FREE tier (1,500 req/day), supports structured output

    # Pinecone Cloud Vector DB
    PINECONE_API_KEY: str | None = None
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "medical-knowledge"

    # Tavily Search (optional fallback)
    TAVILY_API_KEY: str | None = None

    # PubMed E-utilities API (Phase 4)
    ENTREZ_EMAIL: str = "your_email@example.com"  # Required by NCBI for API access

    # Application Settings
    LOG_LEVEL: str = "INFO"
    SEARCH_RESULTS: int = 5
    USE_CLOUD_VECTOR_DB: bool = True  # True = Pinecone, False = ChromaDB

    # ChromaDB Settings (fallback/local)
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval Settings
    RETRIEVAL_TOP_K: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Reranking Settings
    # Reranker provides +16% recall improvement (0.336→0.390) but adds ~2.1s latency on CPU
    # Trade-off: Quality vs Speed. Enable if you have GPU acceleration or can tolerate latency.
    # See Phase 3 experiments in README for measured results.
    USE_RERANKER: bool = False  # Disabled by default due to CPU latency cost
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    RERANKER_TOP_K_MULTIPLIER: int = 4  # Retrieve 4x candidates for reranking

    # Model configuration for loading from .env or config.env
    model_config = SettingsConfigDict(
        env_file=[".env", "config.env"],
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Global settings instance
settings = Settings()
