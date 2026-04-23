# Elasticsearch MCP Server

A **Model Context Protocol (MCP)** server that exposes Elasticsearch semantic search capabilities as MCP tools. This server is optimized for **Elasticsearch Serverless** and managed Elastic Cloud clusters.

---

## 🚀 Features

- **Semantic Search**: Use ELSER (sparse) or kNN (dense) directly within Elasticsearch.
- **Hybrid Search**: Combine keyword (BM25) and semantic scores.
- **Automatic Setup**: Tools to create indices with the correct mappings and ingest pipelines in one click.
- **Serverless Ready**: Fully compatible with Elasticsearch Serverless and API Key authentication.

---

## 🛠️ Quickstart

### 1. Configure Environment
Edit the `.env` file with your connection details:
```env
ELASTICSEARCH_HOSTS=https://your-serverless-endpoint.es.us-east-1.aws.elastic.cloud:443
ELASTICSEARCH_API_KEY=your_api_key
```

### 2. Verify Connection
Run the verification suite to ensure your cluster is reachable and tools are registered:
```powershell
.\venv\Scripts\python.exe verify_server.py
```

### 2. Run the Web API (Standard REST)
Start the web server with a simple command:
```powershell
elasticsearch-api
```
- **API URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`

### 3. Run the MCP Server (For AI Editors)
Start the MCP server with a simple command:
```powershell
elasticsearch-mcp
```

---

## 🧩 Claude Desktop Integration

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "elasticsearch": {
      "command": "C:\\Users\\birar\\Desktop\\elasticsearch-mcp-server\\venv\\Scripts\\python.exe",
      "args": ["-m", "elasticsearch_mcp"],
      "env": {
        "ELASTICSEARCH_HOSTS": "https://your-cluster-url.es.aws.elastic.cloud:443",
        "ELASTICSEARCH_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

---

## 🛠️ Available Tools

- `es_ping`: Check connectivity.
- `es_setup_elser_index`: Create a semantic index for ELSER.
- `es_setup_dense_index`: Create a semantic index for dense vectors (kNN).
- `es_index_document`: Index data through a pipeline.
- `es_semantic_search_elser`: Perform sparse semantic search.
- `es_semantic_search_knn`: Perform dense kNN search.
- `es_semantic_search_hybrid`: Combined keyword and semantic search.
- `es_delete_index`: Safely remove indices.

---

## ⚠️ Requirements
- **Python 3.11+**
- **Elasticsearch 8.8+** (including Serverless)
- **ML Models**: ELSER or E5 must be available/deployed in your cluster for semantic search tools to function.