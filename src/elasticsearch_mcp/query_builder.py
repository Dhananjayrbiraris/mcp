"""Query builders for Elasticsearch semantic search strategies."""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Sparse vector / ELSER (text_expansion)
# ---------------------------------------------------------------------------

def build_elser_query(
    query_text: str,
    field: str,
    model_id: str = ".elser_model_2",
    boost: float = 1.0,
    filter_: Optional[dict] = None,
    source_fields: Optional[list[str]] = None,
    highlight_field: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build a sparse-vector semantic search query using ELSER.

    Uses the `text_expansion` query which applies the ELSER model at query
    time to expand the text into weighted tokens.
    """
    query: dict[str, Any] = {
        "text_expansion": {
            field: {
                "model_id": model_id,
                "model_text": query_text,
                "boost": boost,
            }
        }
    }

    body: dict[str, Any] = {"query": query}

    if filter_:
        body["query"] = {
            "bool": {
                "must": [query],
                "filter": [filter_],
            }
        }

    _apply_common_options(body, source_fields, highlight_field, field)
    return body


# ---------------------------------------------------------------------------
# Dense vector / kNN (e5, multilingual-e5, etc.)
# ---------------------------------------------------------------------------

def build_knn_query(
    query_text: str,
    field: str,
    model_id: str = ".multilingual-e5-small",
    k: int = 10,
    num_candidates: int = 100,
    filter_: Optional[dict] = None,
    source_fields: Optional[list[str]] = None,
    boost: float = 1.0,
) -> dict[str, Any]:
    """
    Build a dense-vector kNN semantic search query.

    Elasticsearch generates the query embedding at search time using the
    `query_vector_builder` with the specified model – no external LLM needed.
    """
    knn_clause: dict[str, Any] = {
        "field": field,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": model_id,
                "model_text": query_text,
            }
        },
        "k": k,
        "num_candidates": num_candidates,
        "boost": boost,
    }

    if filter_:
        knn_clause["filter"] = filter_

    body: dict[str, Any] = {"knn": knn_clause}
    if source_fields:
        body["_source"] = source_fields

    return body


# ---------------------------------------------------------------------------
# Hybrid: ELSER sparse + BM25 keyword (RRF / linear combination)
# ---------------------------------------------------------------------------

def build_hybrid_query(
    query_text: str,
    semantic_field: str,
    keyword_fields: list[str],
    model_id: str = ".elser_model_2",
    semantic_boost: float = 1.0,
    keyword_boost: float = 0.5,
    filter_: Optional[dict] = None,
    source_fields: Optional[list[str]] = None,
    highlight_field: Optional[str] = None,
) -> dict[str, Any]:
    """
    Hybrid query: sparse ELSER semantic + BM25 keyword search.

    Combines both signals via a bool/should query with individual boosts.
    For true RRF re-ranking (ES 8.8+) use build_rrf_query instead.
    """
    semantic_clause: dict[str, Any] = {
        "text_expansion": {
            semantic_field: {
                "model_id": model_id,
                "model_text": query_text,
                "boost": semantic_boost,
            }
        }
    }

    keyword_clause: dict[str, Any] = {
        "multi_match": {
            "query": query_text,
            "fields": keyword_fields,
            "boost": keyword_boost,
        }
    }

    bool_query: dict[str, Any] = {
        "bool": {
            "should": [semantic_clause, keyword_clause],
            "minimum_should_match": 1,
        }
    }

    if filter_:
        bool_query["bool"]["filter"] = [filter_]

    body: dict[str, Any] = {"query": bool_query}
    _apply_common_options(body, source_fields, highlight_field, semantic_field)
    return body


# ---------------------------------------------------------------------------
# Hybrid kNN + BM25 with Reciprocal Rank Fusion (ES 8.8+)
# ---------------------------------------------------------------------------

def build_rrf_query(
    query_text: str,
    dense_field: str,
    keyword_fields: list[str],
    model_id: str = ".multilingual-e5-small",
    k: int = 10,
    num_candidates: int = 100,
    rrf_rank_constant: int = 60,
    rrf_window_size: int = 100,
    filter_: Optional[dict] = None,
    source_fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    True hybrid search with Reciprocal Rank Fusion (RRF).

    Requires Elasticsearch 8.8+ with the rank feature. Combines dense kNN
    and keyword BM25 results via RRF without needing manual boost tuning.
    """
    knn_clause: dict[str, Any] = {
        "field": dense_field,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": model_id,
                "model_text": query_text,
            }
        },
        "k": k,
        "num_candidates": num_candidates,
    }
    if filter_:
        knn_clause["filter"] = filter_

    body: dict[str, Any] = {
        "knn": knn_clause,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": keyword_fields,
            }
        },
        "rank": {
            "rrf": {
                "rank_constant": rrf_rank_constant,
                "window_size": rrf_window_size,
            }
        },
    }

    if filter_:
        body["query"] = {
            "bool": {
                "must": [body["query"]],
                "filter": [filter_],
            }
        }

    if source_fields:
        body["_source"] = source_fields

    return body


# ---------------------------------------------------------------------------
# Semantic text field (ES 8.11+ `semantic_text` mapping type)
# ---------------------------------------------------------------------------

def build_semantic_text_query(
    query_text: str,
    field: str,
    filter_: Optional[dict] = None,
    source_fields: Optional[list[str]] = None,
    highlight_field: Optional[str] = None,
) -> dict[str, Any]:
    """
    Query for the native `semantic_text` field type (ES 8.11+).

    The model is specified at mapping time; at query time you simply use
    the `semantic` query type – Elasticsearch handles everything.
    """
    semantic_query: dict[str, Any] = {"semantic": {"field": field, "query": query_text}}

    body: dict[str, Any]
    if filter_:
        body = {
            "query": {
                "bool": {
                    "must": [semantic_query],
                    "filter": [filter_],
                }
            }
        }
    else:
        body = {"query": semantic_query}

    _apply_common_options(body, source_fields, highlight_field, field)
    return body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_common_options(
    body: dict[str, Any],
    source_fields: Optional[list[str]],
    highlight_field: Optional[str],
    primary_field: str,
) -> None:
    if source_fields:
        body["_source"] = source_fields
    if highlight_field:
        body["highlight"] = {
            "fields": {highlight_field: {"number_of_fragments": 3, "fragment_size": 150}}
        }


def parse_hits(response: dict[str, Any], min_score: float = 0.0) -> list[dict[str, Any]]:
    """
    Extract and normalise hits from a raw Elasticsearch search response.

    Returns a clean list of result dicts ready to hand back via MCP.
    """
    hits_wrapper = response.get("hits", {})
    raw_hits: list[dict] = hits_wrapper.get("hits", [])
    total = hits_wrapper.get("total", {})
    total_count = total.get("value", 0) if isinstance(total, dict) else total

    results = []
    for hit in raw_hits:
        score = hit.get("_score") or 0.0
        if score < min_score:
            continue
        result = {
            "_index": hit.get("_index"),
            "_id": hit.get("_id"),
            "_score": score,
            **hit.get("_source", {}),
        }
        if "highlight" in hit:
            result["_highlight"] = hit["highlight"]
        results.append(result)

    return results
