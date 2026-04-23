"""Entry point for the Elasticsearch MCP Server."""

import asyncio
import logging
import sys
import os

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

import mcp.server.stdio
import mcp.types as types

from elasticsearch_mcp.config import ElasticsearchConfig
from elasticsearch_mcp.server import create_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Starting Elasticsearch MCP Server …")
    config = ElasticsearchConfig.from_env()
    logger.info("Connecting to: %s", config.hosts)

    server = create_server(config)

    # Build initialization options — compatible with MCP >= 1.0
    init_options = server.create_initialization_options()

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("MCP stdio transport ready. Waiting for client …")
        await server.run(
            read_stream,
            write_stream,
            init_options,
        )


if __name__ == "__main__":
    asyncio.run(main())
