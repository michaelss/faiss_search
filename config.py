"""
Configuration settings for the PDF search system.
"""

# Embedding model
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-large"

# PDF processing
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

# FAISS index
INDEX_PATH = "faiss_index"
METADATA_PATH = "documents_metadata.pkl"

# Search
SEARCH_K = 5

# Caminho para os arquivos PDF e URLs
SOURCES = [ 
    "pdfs",
    "https://www.tre-to.jus.br/legislacao/compilada/resolucao/2025/resolucao-no-606-de-26-de-junho-de-2025",
    "https://www.tre-to.jus.br/legislacao/compilada/resolucao/2025/resolucao-no-601-de-14-de-fevereiro-de-2025",
]
