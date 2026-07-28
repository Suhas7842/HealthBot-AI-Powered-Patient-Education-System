"""
Configuration management using Pydantic settings.
Loads environment variables from .env or config.env files.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration (OpenAI-compatible or Gemini)
    LLM_PROVIDER: str = "openai"  # "openai", "groq", or "gemini"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0
    OPENAI_MAX_TOKENS: int = 1000
    OPENAI_BASE_URL: Optional[str] = None  # For Groq: https://api.groq.com/openai/v1

    # Google Gemini Configuration
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash"  # FREE tier, supports structured output

    # Pinecone Cloud Vector DB
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "medical-knowledge"

    # Tavily Search (optional fallback)
    TAVILY_API_KEY: Optional[str] = None

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

    # Model configuration for loading from .env or config.env
    model_config = SettingsConfigDict(
        env_file=[".env", "config.env"],
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
