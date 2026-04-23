"""
verify_server.py  --  End-to-end smoke-test for the Elasticsearch MCP Server.

Tests:
  1. Environment / config loading
  2. Elasticsearch connectivity & cluster info
  3. MCP server creation + tool listing (no ES needed)
  4. Full round-trip: index a doc, search it, delete index

Run:
    python verify_server.py
"""

import asyncio
import json
import os
import sys
import time
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load .env
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from elasticsearch_mcp.config import ElasticsearchConfig
from elasticsearch_mcp.es_client import ElasticsearchClient
from elasticsearch_mcp.server import create_server

# -- Helpers ------------------------------------------------------------------

OK   = "[OK]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
INFO = "[INFO]"

passed = failed = skipped = 0

def check(label, condition, skip=False, detail=""):
    global passed, failed, skipped
    if skip:
        skipped += 1
        print(f"  {SKIP} {label} (skipped)")
    elif condition:
        passed += 1
        print(f"  {OK} {label}")
        if detail:
            print(f"       {detail}")
    else:
        failed += 1
        print(f"  {FAIL} {label}")
        if detail:
            print(f"       {detail}")

# -- Tests --------------------------------------------------------------------

async def test_config():
    print("\n-- 1. Config loading ----------------------------------")
    cfg = ElasticsearchConfig.from_env()
    check("ELASTICSEARCH_HOSTS set", bool(cfg.hosts), detail=str(cfg.hosts))
    check("Timeout > 0",   cfg.request_timeout > 0, detail=f"{cfg.request_timeout}s")
    check("Default model", bool(cfg.default_model_id), detail=cfg.default_model_id)
    return cfg


async def test_connectivity(cfg):
    print("\n-- 2. Elasticsearch connectivity ----------------------")
    es = ElasticsearchClient(cfg)
    alive = False
    try:
        alive = await es.ping()
        check("Ping cluster", alive,
              detail=f"Host: {cfg.hosts[0]}" if alive else "Cannot reach Elasticsearch")

        if alive:
            info = await es.cluster_info()
            check("Cluster info returned", bool(info))
            version = info.get("version", {}).get("number", "?")
            major = version.split(".")[0] if version != "?" else "0"
            check("ES version 8.x+", int(major) >= 8,
                  detail=f"Version: {version}")
            name = info.get("cluster_name", "?")
            print(f"  {INFO} Cluster name: {name}")
    finally:
        await es.close()
    return alive


async def test_mcp_tools():
    print("\n-- 3. MCP server & tool list --------------------------")
    cfg  = ElasticsearchConfig.from_env()
    srv  = create_server(cfg)

    # Trigger list_tools handler
    from elasticsearch_mcp.server import _build_tool_list
    tools = _build_tool_list()
    check("Tools list non-empty", len(tools) > 0, detail=f"{len(tools)} tools registered")

    expected = [
        "es_ping", "es_list_indices", "es_get_mapping", "es_list_ml_models",
        "es_model_stats", "es_semantic_search_elser", "es_semantic_search_knn",
        "es_semantic_search_hybrid", "es_semantic_search_rrf",
        "es_semantic_search_native", "es_setup_elser_index", "es_setup_dense_index",
        "es_create_ingest_pipeline", "es_get_ingest_pipeline",
        "es_index_document", "es_bulk_index", "es_raw_search", "es_delete_index",
    ]
    tool_names = {t.name for t in tools}
    missing = [n for n in expected if n not in tool_names]
    check("All expected tools present", not missing,
          detail=f"Missing: {missing}" if missing else f"All {len(expected)} tools present")


async def test_roundtrip(cfg):
    print("\n-- 4. Full round-trip (index -> search -> delete) -----")
    es   = ElasticsearchClient(cfg)
    idx  = "mcp-verify-test"

    try:
        # Ping first
        if not await es.ping():
            check("Round-trip (ES not reachable)", False, skip=True)
            return

        # Create a simple keyword index (no ML model needed)
        resp = await es.create_index(idx, {
            "properties": {
                "content": {"type": "text"},
                "tag":     {"type": "keyword"},
            }
        })
        check("Create index", resp.get("acknowledged") or resp.get("index") == idx)

        # Index a document
        doc_resp = await es.index_document(idx, {"content": "Hello from MCP verify", "tag": "test"})
        check("Index document", doc_resp.get("result") == "created", detail=str(doc_resp.get("result")))

        # Refresh so doc is searchable
        await es.client.indices.refresh(index=idx)

        # Raw search
        result = await es.execute_search(idx, {"query": {"match": {"content": "MCP"}}}, size=5)
        hits = result.get("hits", {}).get("hits", [])
        check("Search returns hit", len(hits) > 0, detail=f"{len(hits)} hit(s)")

        # Delete index
        del_resp = await es.delete_index(idx)
        check("Delete index", del_resp.get("acknowledged", False))

    except Exception as exc:
        check(f"Round-trip error: {exc}", False)
    finally:
        await es.close()


async def test_ml_models(cfg):
    print("\n-- 5. ML model listing --------------------------------")
    es = ElasticsearchClient(cfg)
    try:
        if not await es.ping():
            check("ML models (ES not reachable)", False, skip=True)
            return
        models = await es.get_ml_models()
        check("ML models API responds", isinstance(models, list))
        if models:
            ids = [m.get("model_id") for m in models]
            print(f"  {INFO} Models found: {ids}")
            has_elser = any(".elser" in (m.get("model_id") or "") for m in models)
            check("ELSER model present", has_elser,
                  skip=not has_elser,
                  detail="Deploy .elser_model_2 in Kibana ML -> Trained Models")
        else:
            print(f"  {INFO} No ML models deployed yet -- deploy ELSER/E5 before semantic search")
    finally:
        await es.close()


# -- Summary ------------------------------------------------------------------

async def main():
    print("=" * 58)
    print("  Elasticsearch MCP Server  --  Verification Suite")
    print("=" * 58)

    cfg   = await test_config()
    alive = await test_connectivity(cfg)
    await test_mcp_tools()

    if alive:
        await test_roundtrip(cfg)
        await test_ml_models(cfg)

    print()
    print("=" * 58)
    total = passed + failed + skipped
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped  ({total} total)")

    if not alive:
        print()
        print("  [!] Elasticsearch is not running.")
        print("  Start it with:  .\\setup_elasticsearch.ps1")
        print("  Or connect to Elastic Cloud by setting ELASTICSEARCH_HOSTS")
        print("  and ELASTICSEARCH_API_KEY in your .env file.")

    if failed == 0:
        print()
        print("  [OK] Server is ready!  Start with:")
        print("    .\\venv\\Scripts\\python.exe -m elasticsearch_mcp")
    print("=" * 58)
    print()

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
