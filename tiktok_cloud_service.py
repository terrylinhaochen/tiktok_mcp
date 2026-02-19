import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, List, Tuple

# MCP imports
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI
import uvicorn

# Import the client
from tiktok_mcp_service.tiktok_client import TikTokClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TikTok client
tiktok_client = TikTokClient()

# Initialize MCP app with lifespan
mcp = FastMCP(
    name="TikTok MCP Service",
    description="A Model Context Protocol service for searching TikTok videos",
    version="1.0.0"
)

# Create a FastAPI app for HTTP transport
http_app = FastAPI(title="TikTok MCP Service")

# Wire up MCP HTTP transport
mcp.wire_fastapi(http_app)

# Set up root endpoint for health check
@http_app.get("/")
async def root():
    return {
        "status": "running",
        "service": "TikTok MCP Service",
        "version": "1.0.0"
    }

# Include your existing MCP endpoints here
# ...

if __name__ == "__main__":
    # Check if running in cloud environment
    PORT = int(os.environ.get("PORT", 8000))
    
    # Start the HTTP server
    logger.info(f"Starting TikTok MCP Service on port {PORT}")
    uvicorn.run(http_app, host="0.0.0.0", port=PORT) 