"""Configuration management for Elasticsearch MCP Server."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ElasticsearchConfig:
    """Elasticsearch connection and behavior configuration."""

    # Connection
    hosts: list[str] = field(default_factory=lambda: ["http://localhost:9200"])
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    ca_certs: Optional[str] = None
    verify_certs: bool = True

    # Timeouts (seconds)
    request_timeout: int = 120
    max_retries: int = 3

    # Semantic Search defaults
    default_model_id: str = ".elser_model_2"          # ELSER v2 (sparse)
    default_dense_model_id: str = ".multilingual-e5-small"  # Dense model
    default_top_k: int = 10
    default_num_candidates: int = 100                  # kNN candidates
    default_min_score: float = 0.0

    @classmethod
    def from_env(cls) -> "ElasticsearchConfig":
        """Load configuration from environment variables."""
        hosts_raw = os.getenv("ELASTICSEARCH_HOSTS", "http://localhost:9200")
        hosts = [h.strip() for h in hosts_raw.split(",")]

        return cls(
            hosts=hosts,
            username=os.getenv("ELASTICSEARCH_USERNAME"),
            password=os.getenv("ELASTICSEARCH_PASSWORD"),
            api_key=os.getenv("ELASTICSEARCH_API_KEY"),
            ca_certs=os.getenv("ELASTICSEARCH_CA_CERTS"),
            verify_certs=os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true",
            request_timeout=int(os.getenv("ELASTICSEARCH_TIMEOUT", "30")),
            max_retries=int(os.getenv("ELASTICSEARCH_MAX_RETRIES", "3")),
            default_model_id=os.getenv("ELASTICSEARCH_DEFAULT_MODEL_ID", ".elser_model_2"),
            default_dense_model_id=os.getenv(
                "ELASTICSEARCH_DENSE_MODEL_ID", ".multilingual-e5-small"
            ),
            default_top_k=int(os.getenv("ELASTICSEARCH_DEFAULT_TOP_K", "10")),
            default_num_candidates=int(
                os.getenv("ELASTICSEARCH_NUM_CANDIDATES", "100")
            ),
            default_min_score=float(os.getenv("ELASTICSEARCH_MIN_SCORE", "0.0")),
        )
