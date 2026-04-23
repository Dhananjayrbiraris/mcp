"""
FastAPI Web Service for Elasticsearch Semantic Search.
Provides a standard REST API wrapper around the MCP tool logic.
"""

import logging
from typing import Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from .config import ElasticsearchConfig
from .es_client import ElasticsearchClient
from .server import _dispatch, _build_tool_list

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- State management ---

class AppState:
    es: Optional[ElasticsearchClient] = None
    config: Optional[ElasticsearchConfig] = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    state.config = ElasticsearchConfig.from_env()
    state.es = ElasticsearchClient(state.config)
    logger.info("Connected to Elasticsearch: %s", state.config.hosts)
    yield
    # Shutdown
    if state.es:
        await state.es.close()

app = FastAPI(
    title="Elasticsearch Semantic Search API",
    description="Standard Web API for ELSER, E5, and Hybrid search.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Models ---

class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any]

# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Elasticsearch Semantic Search API",
        "docs": "/docs"
    }

@app.get("/tools")
async def list_available_tools():
    """List all available semantic search tools and their schemas."""
    tools = _build_tool_list()
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema
        }
        for t in tools
    ]

@app.post("/call")
async def call_tool(request: ToolCallRequest):
    """
    Generic endpoint to call any tool by name.
    Example: { "name": "es_ping", "arguments": {} }
    """
    if not state.es or not state.config:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    try:
        result = await _dispatch(request.name, request.arguments, state.es, state.config)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Tool execution failed")
        raise HTTPException(status_code=500, detail=str(e))

# --- Specific Convenience Endpoints ---

@app.get("/ping")
async def ping():
    """Check Elasticsearch connectivity."""
    return await call_tool(ToolCallRequest(name="es_ping", arguments={}))

@app.post("/search/elser")
async def search_elser(
    index: str,
    query: str,
    field: str = "content_embedding",
    size: int = 10
):
    """Convenience endpoint for ELSER sparse search."""
    return await call_tool(ToolCallRequest(
        name="es_semantic_search_elser",
        arguments={"index": index, "query": query, "field": field, "size": size}
    ))

@app.post("/search/knn")
async def search_knn(
    index: str,
    query: str,
    field: str = "content_vector",
    k: int = 10
):
    """Convenience endpoint for dense kNN search."""
    return await call_tool(ToolCallRequest(
        name="es_semantic_search_knn",
        arguments={"index": index, "query": query, "field": field, "k": k}
    ))

def main():
    import uvicorn
    uvicorn.run("elasticsearch_mcp.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
