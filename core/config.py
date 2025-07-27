from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Embedding model
    EMBEDDINGS_MODEL: str = "intfloat/multilingual-e5-large"

    # PDF processing
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 400

    # FAISS index
    INDEX_PATH: str = "faiss_index"
    METADATA_PATH: str = "documents_metadata.pkl"

    # Search
    SEARCH_K: int = 5

    # Caminho para os arquivos PDF e URLs
    SOURCES: List[str] = [
        "pdfs", "txts",
        "https://www.tre-to.jus.br/legislacao/compilada/resolucao/2025/resolucao-no-606-de-26-de-junho-de-2025",
        "https://www.tre-to.jus.br/legislacao/compilada/resolucao/2025/resolucao-no-601-de-14-de-fevereiro-de-2025",
    ]

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4o-mini"
    AZURE_OPENAI_API_VERSION: str = "2024-08-01-preview"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()