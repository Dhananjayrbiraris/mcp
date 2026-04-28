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
    using Elasticsearch's native 'semantic' query type.
    """
    
    def search_semantic(self, index_name, query_text, field="description", size=5):
        """
        Performs a native semantic search using the es_raw_search tool.
        """
        body = {
            "query": {
                "semantic": {
                    "field": field,
                    "query": query_text
                }
            }
        }
        return call_tool("es_raw_search", {
            "index": index_name,
            "body": body,
            "size": size
        })
        
    def get_documents(self, index_name, size=20):
        """
        Gets raw documents from an index.
        """
        print(f"Fetching {size} documents from '{index_name}'...")
        body = {
            "query": {"match_all": {}}
        }
        return call_tool("es_raw_search", {
            "index": index_name,
            "body": body,
            "size": size
        })

# --- Main Example ---

if __name__ == "__main__":
    searcher = MultiIndexSearcher()
    
    # # 1. Get raw documents from company_details
    # print("\n--- TEST 1: Get Company Details ---")
    # results1 = searcher.get_documents("company_details", size=2) # size 2 for brevity in console
    # if results1 and "hits" in results1 and "hits" in results1["hits"]:
    #     hits_list = results1["hits"]["hits"]
    #     print(f"Found {len(hits_list)} results.")
    #     if hits_list:
    #         print(f"First hit: {json.dumps(hits_list[0].get('_source', {}), indent=2)}")
    # else:
    #     print("Could not retrieve documents. Is the index correct?")

    # 2. Semantic Search on om_poc_v2
    print("\n--- TEST 2: Semantic Search (om_poc_v2) ---")
    queries = [
        "I want to exercise at home and recover from workouts without going to the gym.",
        "I work on a computer all day and my neck and back hurt. What can help improve my posture?",
        "I want to stay hydrated and track my daily water intake.",
        "My eyes hurt after staring at screens. What can help reduce eye strain?",
        "I want to keep my desk clean and organized while working from home."
    ]
    
    for q in queries:
        print(f"\n-> Query: {q}")
        res = searcher.search_semantic("om_poc_v2", q, size=5) # Get top 3 matches per query
        if res and "hits" in res and "hits" in res["hits"]:
            hits_list = res["hits"]["hits"]
            if hits_list:
                print(f"   Found {len(hits_list)} matches:")
                for i, hit in enumerate(hits_list, start=1):
                    source = hit.get('_source', {})
                    print(f"   [{i}] Score: {hit.get('_score')} | Name: {source.get('name', 'N/A')} | Price: ${source.get('price', 'N/A')}")
                    print(f"       Desc : {str(source.get('description', ''))[:80]}...")
            else:
                print("   No results found.")
        else:
            print("   No results found or an error occurred.")
