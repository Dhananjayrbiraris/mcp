"""Elasticsearch client wrapper with connection management."""

import logging
from typing import Any, Optional

from elasticsearch import AsyncElasticsearch, NotFoundError, RequestError

from .config import ElasticsearchConfig

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Async Elasticsearch client with helper methods for semantic search."""

    def __init__(self, config: ElasticsearchConfig):
        self.config = config
        self._client: Optional[AsyncElasticsearch] = None

    def _build_client(self) -> AsyncElasticsearch:
        """Build the AsyncElasticsearch client from config."""
        kwargs: dict[str, Any] = {
            "hosts": self.config.hosts,
            "request_timeout": self.config.request_timeout,
            "max_retries": self.config.max_retries,
            "retry_on_timeout": True,
        }

        # Auth: API key takes priority over basic auth
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        elif self.config.username and self.config.password:
            kwargs["basic_auth"] = (self.config.username, self.config.password)

        # TLS
        if self.config.ca_certs:
            kwargs["ca_certs"] = self.config.ca_certs
        if not self.config.verify_certs:
            kwargs["verify_certs"] = False
            kwargs["ssl_show_warn"] = False

        return AsyncElasticsearch(**kwargs)

    @property
    def client(self) -> AsyncElasticsearch:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def ping(self) -> bool:
        """Check cluster connectivity."""
        try:
            return await self.client.ping()
        except Exception as exc:
            logger.warning("Elasticsearch ping failed: %s", exc)
            return False

    async def cluster_info(self) -> dict[str, Any]:
        """Return basic cluster info."""
        info = await self.client.info()
        return dict(info)

    async def list_indices(self, pattern: str = "*") -> list[dict[str, Any]]:
        """List indices matching a pattern."""
        response = await self.client.cat.indices(
            index=pattern, h="index,health,status,docs.count,store.size", format="json"
        )
        return list(response)

    async def get_mapping(self, index: str) -> dict[str, Any]:
        """Return index mapping."""
        resp = await self.client.indices.get_mapping(index=index)
        return dict(resp)

    async def get_ml_models(self) -> list[dict[str, Any]]:
        """List deployed ML models available for inference."""
        try:
            try:
                resp = await self.client.ml.get_trained_models(
                    model_id="*", include="definition_status"
                )
            except Exception:
                # ES 9.x does not support include="definition_status"
                resp = await self.client.ml.get_trained_models(model_id="*")
            models = resp.get("trained_model_configs", [])
            return [
                {
                    "model_id": m.get("model_id"),
                    "model_type": m.get("model_type"),
                    "tags": m.get("tags", []),
                    "state": m.get("fully_defined", False),
                }
                for m in models
            ]
        except Exception as exc:
            logger.warning("Could not fetch ML models: %s", exc)
            return []

    async def model_deployment_stats(self, model_id: str) -> dict[str, Any]:
        """Get deployment stats for a specific ML model."""
        try:
            resp = await self.client.ml.get_trained_models_stats(model_id=model_id)
            configs = resp.get("trained_model_stats", [])
            return dict(configs[0]) if configs else {}
        except NotFoundError:
            return {"error": f"Model '{model_id}' not found"}
        except Exception as exc:
            return {"error": str(exc)}

    async def execute_search(
        self, index: str, body: dict[str, Any], size: int = 10
    ) -> dict[str, Any]:
        """Execute a raw search query."""
        resp = await self.client.search(index=index, body=body, size=size)
        return dict(resp)

    async def index_document(
        self,
        index: str,
        document: dict[str, Any],
        doc_id: Optional[str] = None,
        pipeline: Optional[str] = None,
    ) -> dict[str, Any]:
        """Index a document, optionally through an ingest pipeline."""
        kwargs: dict[str, Any] = {"index": index, "body": document}
        if doc_id:
            kwargs["id"] = doc_id
        if pipeline:
            kwargs["pipeline"] = pipeline
        resp = await self.client.index(**kwargs)
        return dict(resp)

    async def bulk_index(
        self,
        index: str,
        documents: list[dict[str, Any]],
        pipeline: Optional[str] = None,
    ) -> dict[str, Any]:
        """Bulk index documents."""
        from elasticsearch.helpers import async_bulk

        def _gen():
            for doc in documents:
                action: dict[str, Any] = {"_index": index, "_source": doc}
                if pipeline:
                    action["pipeline"] = pipeline
                yield action

        success, errors = await async_bulk(self.client, _gen(), raise_on_error=False)
        return {"indexed": success, "errors": errors}

    async def create_ingest_pipeline(
        self, pipeline_id: str, pipeline_body: dict[str, Any]
    ) -> dict[str, Any]:
        """Create or update an ingest pipeline."""
        resp = await self.client.ingest.put_pipeline(
            id=pipeline_id, body=pipeline_body
        )
        return dict(resp)

    async def get_ingest_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        """Get an ingest pipeline definition."""
        try:
            resp = await self.client.ingest.get_pipeline(id=pipeline_id)
            return dict(resp)
        except NotFoundError:
            return {"error": f"Pipeline '{pipeline_id}' not found"}

    async def create_index(
        self, index: str, mappings: dict[str, Any], settings: Optional[dict] = None
    ) -> dict[str, Any]:
        """Create an index with given mappings."""
        body: dict[str, Any] = {"mappings": mappings}
        if settings:
            body["settings"] = settings
        resp = await self.client.indices.create(index=index, body=body)
        return dict(resp)

    async def delete_index(self, index: str) -> dict[str, Any]:
        """Delete an index."""
        resp = await self.client.indices.delete(index=index)
        return dict(resp)

    async def close(self) -> None:
        """Close the underlying client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
