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
