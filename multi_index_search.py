import requests
import json

# --- Configuration ---
API_URL = "http://localhost:8000"

def call_tool(name, arguments):
    """Helper to call the Elasticsearch Web API."""
    try:
        resp = requests.post(f"{API_URL}/call", json={"name": name, "arguments": arguments})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"API Error ({name}): {e}")
        return None

# --- Dynamic Search Logic ---

class MultiIndexSearcher:
    """
    A helper class to manage searches across different indices 
    with different semantic search requirements.
    """
    
    def __init__(self):
        # Configuration for different indices
        # In a real app, this could be loaded from a database or config file.
        self.index_configs = {
            "demo-products": {
                "type": "knn",
                "field": "text_vector",
                "model_id": ".multilingual-e5-small-elasticsearch",
                "params": {"k": 5}
            },
            "demo-knowledge-base": {
                "type": "elser",
                "field": "content_embedding",
                "params": {"size": 3}
            }
        }

    def search(self, index_name, query_text):
        """
        Dynamically chooses the right search tool and parameters based on the index.
        """
        config = self.index_configs.get(index_name)
        
        if not config:
            print(f"Warning: Index '{index_name}' not configured. Defaulting to raw search.")
            return call_tool("es_raw_search", {
                "index": index_name,
                "body": {"query": {"match": {"text": query_text}}}
            })

        if config["type"] == "knn":
            print(f"Performing Dense kNN search on '{index_name}' using E5...")
            return call_tool("es_semantic_search_knn", {
                "index": index_name,
                "query": query_text,
                "field": config["field"],
                "model_id": config["model_id"],
                "k": config["params"].get("k", 10)
            })
            
        elif config["type"] == "elser":
            print(f"Performing Sparse ELSER search on '{index_name}'...")
            return call_tool("es_semantic_search_elser", {
                "index": index_name,
                "query": query_text,
                "field": config["field"],
                "size": config["params"].get("size", 10)
            })

# --- Main Example ---

if __name__ == "__main__":
    searcher = MultiIndexSearcher()
    
    # 1. Search in the Product index (Vector/E5)
    print("\n--- TEST 1: Products ---")
    results1 = searcher.search("demo-products", "Wireless noise canceling headphones")
    if results1 and "hits" in results1:
        print(f"Found {len(results1['hits'])} results.")

    # 2. Search in the Knowledge Base index (Sparse/ELSER)
    print("\n--- TEST 2: Knowledge Base ---")
    results2 = searcher.search("demo-knowledge-base", "How do I reset my account password?")
    if results2 and "hits" in results2:
        print(f"Found {len(results2['hits'])} results.")
        
    # 3. Search in an unknown index (Fallback)
    print("\n--- TEST 3: Unknown Index ---")
    searcher.search("random-index", "hello")
