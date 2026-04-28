# Elasticsearch Web API

A FastAPI-based web server that exposes Elasticsearch semantic search and data retrieval capabilities through a simple REST API. This project connects directly to your Elasticsearch cluster (including Elastic Cloud and Serverless) and allows you to build custom search applications effortlessly.

---

## 🚀 Features

- **Semantic Search**: Use native Elasticsearch semantic capabilities directly through a REST API.
- **REST API First**: Fully functional web API with built-in Swagger documentation.
- **Data Retrieval**: Easily pull raw documents from any index.
- **Python Integration**: Includes a helper script (`multi_index_search.py`) demonstrating how to interact with the API using Python.

---

## 🛠️ Quickstart

### 1. Configure Environment
Edit the `.env` file with your Elasticsearch connection details:
```env
ELASTICSEARCH_HOSTS=https://your-cluster-url.es.aws.elastic.cloud:443
ELASTICSEARCH_API_KEY=your_api_key_here
```

### 2. Start the Web API
Start the web server using the pre-configured executable in your virtual environment:
```powershell
.\venv\Scripts\elasticsearch-api.exe
```

Once running, you can access:
- **API Health Check**: `http://localhost:8000/ping`
- **Interactive Documentation (Swagger UI)**: `http://localhost:8000/docs`

### 3. Run the Example Script
A demo script (`multi_index_search.py`) is provided to show how to communicate with the API. While the server is running, open a new terminal and execute:
```powershell
.\venv\Scripts\python.exe multi_index_search.py
```

This script will:
- Connect to the local API via `http://localhost:8000/call`
- Retrieve records from the `company_details` index.
- Perform native semantic searches on the `om_poc_v2` index using natural language queries.
- Print the matched products with their relevance scores, prices, categories, and descriptions!

---

## 🔍 Core Endpoints

- `GET /ping`: Verify your cluster connection.
- `GET /tools`: List all available internal tools.
- `POST /call`: The primary dynamic endpoint. Pass a tool name (like `es_raw_search`) and its arguments (like index and query DSL body) to interact with the Elasticsearch cluster.

---

## ⚠️ Requirements
- **Python 3.11+**
- **Elasticsearch 8.8+** (including Serverless)
- Valid API Keys and a running cluster defined in your `.env` file.