"""
Index and ingest pipeline setup helpers for semantic search.

These helpers generate the correct Elasticsearch index mappings and
ingest pipeline definitions for each supported semantic strategy.
"""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Ingest pipelines
# ---------------------------------------------------------------------------

def elser_ingest_pipeline(
    pipeline_id: str,
    source_field: str,
    target_field: str,
    model_id: str = ".elser_model_2",
) -> tuple[str, dict[str, Any]]:
    """
    Build an ELSER sparse-embedding ingest pipeline.

    The pipeline runs `inference` with the ELSER model during document
    indexing and writes the sparse vector into `target_field`.
    """
    body = {
        "description": f"ELSER sparse embedding pipeline for '{source_field}'",
        "processors": [
            {
                "inference": {
                    "model_id": model_id,
                    "input_output": [
                        {"input_field": source_field, "output_field": target_field}
                    ],
                }
            }
        ],
    }
    return pipeline_id, body


def dense_ingest_pipeline(
    pipeline_id: str,
    source_field: str,
    target_field: str,
    model_id: str = ".multilingual-e5-small",
) -> tuple[str, dict[str, Any]]:
    """
    Build a dense-vector (text-embedding) ingest pipeline.

    The pipeline embeds `source_field` into a float vector stored in
    `target_field`, used later for kNN search.
    """
    body = {
        "description": f"Dense text-embedding pipeline for '{source_field}'",
        "processors": [
            {
                "inference": {
                    "model_id": model_id,
                    "input_output": [
                        {"input_field": source_field, "output_field": target_field}
                    ],
                }
            }
        ],
    }
    return pipeline_id, body


# ---------------------------------------------------------------------------
# Index mappings
# ---------------------------------------------------------------------------

def elser_index_mapping(
    text_field: str,
    sparse_field: str,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Mapping for an index that stores ELSER sparse vectors.

    `text_field`   – keyword/text field holding the raw text.
    `sparse_field` – `sparse_vector` field holding ELSER tokens.
    """
    properties: dict[str, Any] = {
        text_field: {"type": "text"},
        sparse_field: {"type": "sparse_vector"},
    }
    if extra_fields:
        properties.update(extra_fields)
    return {"properties": properties}


def dense_index_mapping(
    text_field: str,
    vector_field: str,
    dims: int,
    similarity: str = "cosine",
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Mapping for an index that stores dense embedding vectors.

    `vector_field` uses `dense_vector` with `index=True` so kNN ANN search
    is enabled.  `similarity` can be `cosine`, `dot_product`, or `l2_norm`.
    """
    properties: dict[str, Any] = {
        text_field: {"type": "text"},
        vector_field: {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": similarity,
        },
    }
    if extra_fields:
        properties.update(extra_fields)
    return {"properties": properties}


def semantic_text_index_mapping(
    semantic_field: str,
    inference_id: str,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Mapping using the native `semantic_text` field type (ES 8.11+).

    `inference_id` references an Elasticsearch inference endpoint.  The
    model selection and vector storage are fully managed by Elasticsearch.
    """
    properties: dict[str, Any] = {
        semantic_field: {
            "type": "semantic_text",
            "inference_id": inference_id,
        }
    }
    if extra_fields:
        properties.update(extra_fields)
    return {"properties": properties}


def hybrid_index_mapping(
    text_field: str,
    sparse_field: str,
    vector_field: str,
    dims: int,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Combined mapping that supports both ELSER sparse and dense kNN search.
    """
    properties: dict[str, Any] = {
        text_field: {"type": "text"},
        sparse_field: {"type": "sparse_vector"},
        vector_field: {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine",
        },
    }
    if extra_fields:
        properties.update(extra_fields)
    return {"properties": properties}


# ---------------------------------------------------------------------------
# Model dimension lookup (common Elasticsearch built-in models)
# ---------------------------------------------------------------------------

KNOWN_MODEL_DIMS: dict[str, int] = {
    ".multilingual-e5-small": 384,
    ".multilingual-e5-small_linux-x86_64": 384,
    "sentence-transformers__all-minilm-l6-v2": 384,
    "sentence-transformers__all-mpnet-base-v2": 768,
    "sentence-transformers__msmarco-minilm-l-12-v3": 384,
}


def get_model_dims(model_id: str) -> Optional[int]:
    """Return known vector dimensions for common ES built-in models."""
    return KNOWN_MODEL_DIMS.get(model_id)
