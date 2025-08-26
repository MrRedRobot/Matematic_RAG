from pydantic_settings import BaseSettings
from typing import Dict

class Settings(BaseSettings):
    """Configuración del sistema RAG optimizada para Ollama"""

    LLM_MODELS: Dict[str, dict] = {
        "ollama_llama3": {
            "model_name": "llama3.2:3b",
            "temperature": 0.1,
            "max_tokens": 2000,
            "context_window": 8192
        },
        "ollama_phi3": {
            "model_name": "phi3:mini",
            "temperature": 0.1,
            "max_tokens": 1500,
            "context_window": 4096
        },
        "ollama_qwen": {
            "model_name": "qwen2.5:3b",
            "temperature": 0.1,
            "max_tokens": 2000,
            "context_window": 8192
        },
        "huggingface_fallback": {
            "model_name": "microsoft/DialoGPT-medium",
            "temperature": 0.7,
            "max_tokens": 100,
            "context_window": 1024
        }
    }

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Streamlit
    STREAMLIT_PORT: int = 8501

    # Chroma
    CHROMA_TELEMETRY_ENABLED: bool = False

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    VECTOR_DB_PATH: str = "vector_store"
    COLLECTION_NAME: str = "math_documents"

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    MAX_CHUNKS_RETRIEVED: int = 3
    MAX_CONTEXT_TOKENS: int = 4000
    MAX_INPUT_TOKENS: int = 6000

    llm_model_type: str = "ollama"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    log_level: str = "INFO"
    log_file: str = "logs/rag_system.log"

    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 120

    class Config:
        env_file = ".env"
        extra = "forbid"

settings = Settings()