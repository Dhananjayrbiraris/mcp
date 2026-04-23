import requests
import json
import time

# Use the Web API we just created
API_URL = "http://localhost:8000"

def call_tool(name, arguments):
    resp = requests.post(f"{API_URL}/call", json={"name": name, "arguments": arguments})
    if resp.status_code != 200:
        print(f"Error calling {name}: {resp.text}")
        return None
    return resp.json()

def run_demo():
    print("--- 1. Checking Connection ---")
    ping = call_tool("es_ping", {})
    if not ping or ping.get("status") != "connected":
        print("Could not connect to the API. Make sure run_web_api.bat is running!")
        return

    index_name = "demo-semantic-search"
    print(f"\n--- 2. Setting up Index: {index_name} ---")
    # Using the existing inference ID from your cluster
    model_id = ".multilingual-e5-small-elasticsearch"
    
    setup = call_tool("es_setup_dense_index", {
        "index": index_name,
        "text_field": "text",
        "vector_field": "text_vector",
        "model_id": model_id,
        "dims": 384, # E5 Small is 384 dims
        "pipeline_id": "demo-pipeline"
    })
    
    if not setup:
        print("Setup failed. Check if ELSER v2 is deployed in your cluster.")
        return
    print("Index and Pipeline created successfully.")

    print("\n--- 3. Indexing Documents ---")
    docs = [
        {"text": "Elasticsearch is a distributed, RESTful search and analytics engine."},
        {"text": "ELSER is a learned sparse encoder that provides high-relevance semantic search."},
        {"text": "Vector search allows you to find documents based on similarity rather than keywords."},
        {"text": "Python is a popular language for data science and AI applications."}
    ]
    
    # Bulk index using the pipeline created in step 2
    bulk = call_tool("es_bulk_index", {
        "index": index_name,
        "documents": docs,
        "pipeline": "demo-pipeline"
    })
    print(f"Indexed {bulk.get('indexed', 0)} documents.")

    # Wait a moment for indexing
    print("Waiting for indexing to complete...")
    time.sleep(2)

    print("\n--- 4. Performing Semantic Search ---")
    query = "How can I find relevant documents using AI?"
    print(f"Query: '{query}'")
    
    results = call_tool("es_semantic_search_knn", {
        "index": index_name,
        "query": query,
        "field": "text_vector",
        "model_id": model_id,
        "k": 3
    })

    if results and "hits" in results:
        print(f"\nFound {len(results['hits'])} results:")
        for i, hit in enumerate(results['hits'], 1):
            print(f"{i}. [Score: {hit['_score']:.4f}] {hit['text']}")
    else:
        print("No results found.")

    # Optional: cleanup
    # print(f"\n--- 5. Cleaning up ---")
    # call_tool("es_delete_index", {"index": index_name, "confirm": True})

if __name__ == "__main__":
    run_demo()
