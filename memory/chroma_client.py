# memory/chroma_client.py

import chromadb
from chromadb.utils import embedding_functions
from typing import Optional

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384
CHROMA_PATH     = "memory/chroma"


def _build_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """
    Explicitly constructs and validates the SentenceTransformer embedding function.
    Never relies on ChromaDB's default embedding — that path leads to silent random embeddings.
    Raises RuntimeError with actionable instructions if the model is not available.
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Validate the model actually loaded by running a test embedding
    try:
        test_result = ef(["embedding validation probe"])
        assert len(test_result) == 1, "Expected exactly 1 embedding vector"
        assert len(test_result[0]) == EMBEDDING_DIM, (
            f"Expected embedding dim {EMBEDDING_DIM}, got {len(test_result[0])}. "
            f"Wrong model loaded or model is corrupted."
        )
    except Exception as e:
        raise RuntimeError(
            f"\n[ChromaDB] Embedding model '{EMBEDDING_MODEL}' failed validation.\n"
            f"Error: {e}\n\n"
            f"Fix: Pre-download the model before running Professor:\n"
            f"  python -c \"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('{EMBEDDING_MODEL}')\"\n"
            f"This downloads ~80MB from HuggingFace. Run once per machine."
        ) from e

    print(f"[ChromaDB] Embedding model verified: {EMBEDDING_MODEL} ({EMBEDDING_DIM}-dim)")
    return ef


def build_chroma_client(persist_dir: str = CHROMA_PATH) -> chromadb.ClientAPI:
    """
    Returns a PersistentClient with a validated embedding function.
    Call this once at startup. Store the result — do not call on every query.
    """
    ef = _build_embedding_function()
    client = chromadb.PersistentClient(path=persist_dir)

    # Attach the validated embedding function to the client for downstream use
    client._professor_ef = ef

    return client


def get_or_create_collection(
    client: chromadb.ClientAPI,
    name: str,
) -> chromadb.Collection:
    """
    Gets or creates a named collection using the validated embedding function.
    Always call this instead of client.get_or_create_collection() directly —
    that path does not guarantee the correct embedding function is used.
    """
    ef = getattr(client, "_professor_ef", None)
    if ef is None:
        raise RuntimeError(
            "[ChromaDB] Client was not created via build_chroma_client(). "
            "Embedding function is missing. Fix: use build_chroma_client()."
        )
    return client.get_or_create_collection(name=name, embedding_function=ef)
