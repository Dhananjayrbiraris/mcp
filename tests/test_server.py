"""Unit tests for the Elasticsearch MCP server."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from elasticsearch_mcp.config import ElasticsearchConfig
from elasticsearch_mcp.query_builder import (
    build_elser_query,
    build_knn_query,
    build_hybrid_query,
    build_rrf_query,
    build_semantic_text_query,
    parse_hits,
)
from elasticsearch_mcp.index_setup import (
    elser_index_mapping,
    dense_index_mapping,
    elser_ingest_pipeline,
    dense_ingest_pipeline,
    get_model_dims,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = ElasticsearchConfig()
        assert cfg.hosts == ["http://localhost:9200"]
        assert cfg.default_model_id == ".elser_model_2"
        assert cfg.default_top_k == 10

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("ELASTICSEARCH_HOSTS", "https://es1:9200,https://es2:9200")
        monkeypatch.setenv("ELASTICSEARCH_USERNAME", "user")
        monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "pass")
        monkeypatch.setenv("ELASTICSEARCH_DEFAULT_TOP_K", "20")
        cfg = ElasticsearchConfig.from_env()
        assert cfg.hosts == ["https://es1:9200", "https://es2:9200"]
        assert cfg.username == "user"
        assert cfg.default_top_k == 20


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

class TestElserQuery:
    def test_basic(self):
        body = build_elser_query("cat food", "content_embedding")
        q = body["query"]["text_expansion"]["content_embedding"]
        assert q["model_text"] == "cat food"
        assert q["model_id"] == ".elser_model_2"

    def test_with_filter(self):
        f = {"term": {"category": "food"}}
        body = build_elser_query("cat food", "content_embedding", filter_=f)
        assert "bool" in body["query"]
        assert body["query"]["bool"]["filter"] == [f]

    def test_with_source_fields(self):
        body = build_elser_query("query", "field", source_fields=["title", "body"])
        assert body["_source"] == ["title", "body"]

    def test_highlight(self):
        body = build_elser_query("query", "field", highlight_field="body")
        assert "highlight" in body
        assert "body" in body["highlight"]["fields"]


class TestKnnQuery:
    def test_basic(self):
        body = build_knn_query("semantic search", "content_vector")
        knn = body["knn"]
        assert knn["field"] == "content_vector"
        assert knn["k"] == 10
        assert knn["query_vector_builder"]["text_embedding"]["model_text"] == "semantic search"

    def test_custom_k(self):
        body = build_knn_query("query", "vec", k=5, num_candidates=50)
        assert body["knn"]["k"] == 5
        assert body["knn"]["num_candidates"] == 50


class TestHybridQuery:
    def test_basic(self):
        body = build_hybrid_query("search", "sparse_field", ["title", "body"])
        assert "bool" in body["query"]
        should = body["query"]["bool"]["should"]
        # Should have both sparse and multi_match clauses
        types_in_should = [list(clause.keys())[0] for clause in should]
        assert "text_expansion" in types_in_should
        assert "multi_match" in types_in_should

    def test_boosts(self):
        body = build_hybrid_query(
            "q", "sf", ["f"], semantic_boost=2.0, keyword_boost=0.3
        )
        should = body["query"]["bool"]["should"]
        sparse = next(c for c in should if "text_expansion" in c)
        kw = next(c for c in should if "multi_match" in c)
        assert sparse["text_expansion"]["sf"]["boost"] == 2.0
        assert kw["multi_match"]["boost"] == 0.3


class TestRrfQuery:
    def test_structure(self):
        body = build_rrf_query("query", "vec_field", ["title"])
        assert "knn" in body
        assert "query" in body
        assert "rank" in body
        assert "rrf" in body["rank"]

    def test_rrf_params(self):
        body = build_rrf_query(
            "q", "v", ["f"], rrf_rank_constant=30, rrf_window_size=50
        )
        rrf = body["rank"]["rrf"]
        assert rrf["rank_constant"] == 30
        assert rrf["window_size"] == 50


class TestSemanticTextQuery:
    def test_basic(self):
        body = build_semantic_text_query("hello world", "my_semantic_field")
        assert body["query"]["semantic"]["field"] == "my_semantic_field"
        assert body["query"]["semantic"]["query"] == "hello world"


# ---------------------------------------------------------------------------
# parse_hits
# ---------------------------------------------------------------------------

class TestParseHits:
    def _make_response(self, hits):
        return {
            "hits": {
                "total": {"value": len(hits)},
                "hits": hits,
            }
        }

    def test_basic_parse(self):
        resp = self._make_response([
            {"_index": "idx", "_id": "1", "_score": 1.5, "_source": {"title": "Hello"}},
        ])
        results = parse_hits(resp)
        assert len(results) == 1
        assert results[0]["title"] == "Hello"
        assert results[0]["_score"] == 1.5

    def test_min_score_filter(self):
        resp = self._make_response([
            {"_index": "i", "_id": "1", "_score": 0.5, "_source": {}},
            {"_index": "i", "_id": "2", "_score": 1.5, "_source": {}},
        ])
        results = parse_hits(resp, min_score=1.0)
        assert len(results) == 1
        assert results[0]["_id"] == "2"

    def test_highlight_included(self):
        resp = self._make_response([
            {
                "_index": "i", "_id": "1", "_score": 1.0,
                "_source": {"title": "x"},
                "highlight": {"title": ["<em>x</em>"]},
            }
        ])
        results = parse_hits(resp)
        assert "_highlight" in results[0]


# ---------------------------------------------------------------------------
# Index setup helpers
# ---------------------------------------------------------------------------

class TestIndexSetup:
    def test_elser_mapping(self):
        mapping = elser_index_mapping("content", "content_embedding")
        props = mapping["properties"]
        assert props["content"]["type"] == "text"
        assert props["content_embedding"]["type"] == "sparse_vector"

    def test_dense_mapping(self):
        mapping = dense_index_mapping("content", "content_vector", dims=384)
        props = mapping["properties"]
        assert props["content_vector"]["type"] == "dense_vector"
        assert props["content_vector"]["dims"] == 384
        assert props["content_vector"]["index"] is True

    def test_elser_pipeline(self):
        pid, body = elser_ingest_pipeline("my-pipeline", "content", "content_embedding")
        assert pid == "my-pipeline"
        proc = body["processors"][0]["inference"]
        assert proc["model_id"] == ".elser_model_2"
        assert proc["input_output"][0]["input_field"] == "content"

    def test_dense_pipeline(self):
        pid, body = dense_ingest_pipeline("dense-pl", "body", "body_vector", ".multilingual-e5-small")
        assert pid == "dense-pl"
        proc = body["processors"][0]["inference"]
        assert proc["model_id"] == ".multilingual-e5-small"

    def test_known_model_dims(self):
        assert get_model_dims(".multilingual-e5-small") == 384
        assert get_model_dims("unknown-model") is None
