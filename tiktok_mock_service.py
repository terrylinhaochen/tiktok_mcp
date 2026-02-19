import os
import sys
import json
import random
import logging
import asyncio
from typing import Dict, Any, List, Tuple

# MCP imports
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP app
mcp = FastMCP(
    name="TikTok Mock Service",
    description="A Model Context Protocol service for simulated TikTok data",
    version="1.0.0"
)

@mcp.resource("status://health")
async def get_health_status() -> Tuple[str, str]:
    """Get the current health status of the service"""
    status = {
        "status": "running",
        "service": {
            "name": "TikTok Mock Service",
            "version": "1.0.0",
            "description": "Simulated TikTok data for Claude"
        }
    }
    return json.dumps(status, indent=2), "application/json"

def generate_mock_videos(term, count=30):
    """Generate mock videos for a search term"""
    # Clean term for use in descriptions
    clean_term = term.lstrip('#').lower()
    
    # Choose relevant topic content
    topic_content = {
        "tariff": "discussing the economic impact of new tariffs on global trade",
        "trump": "covering Trump's latest campaign events and policy announcements",
        "biden": "analyzing President Biden's recent decisions and policy changes",
        "viral": "that's getting millions of views across TikTok right now",
        "cooking": "showing a delicious recipe that's easy to make at home",
    }.get(clean_term, f"about {clean_term} that's trending on TikTok")
    
    # Create usernames based on topic
    usernames_by_topic = {
        "tariff": ["economistDaily", "tradeExpert", "globalEcon", "marketWatcher"],
        "trump": ["politicalAnalyst", "conservativeVoice", "newsUpdates", "politicsToday"],
        "biden": ["whitehouseReporter", "politicalNews", "dcInsider", "politicsTalk"],
        "viral": ["trendingNow", "viralCreator", "tiktokStar", "contentKing"],
        "default": ["user123456", "creator789", "tiktokUser", "contentCreator"]
    }
    
    # Choose relevant usernames or default ones
    usernames = usernames_by_topic.get("default")
    for topic, names in usernames_by_topic.items():
        if topic in clean_term:
            usernames = names
            break
    
    # Generate videos
    videos = []
    for i in range(min(count, 10)):
        username = random.choice(usernames)
        video_id = f"734{random.randint(1000000, 9999999)}"
        
        videos.append({
            'url': f"https://www.tiktok.com/@{username}/video/{video_id}",
            'description': f"#{clean_term} video {topic_content} #viral #trending",
            'author': username.replace("_", " ").title(),
            'stats': {
                'views': str(random.randint(500000, 5000000)),
                'likes': str(random.randint(50000, 500000)),
                'shares': str(random.randint(5000, 50000)),
                'comments': str(random.randint(1000, 10000))
            }
        })
    
    return videos

@mcp.tool()
async def search_videos(search_terms: List[str], count: int = 30) -> Dict[str, Any]:
    """Search for TikTok videos based on search terms"""
    results = {}
    transformations = {}
    
    # Limit count to reasonable number
    count = min(count, 15)
    
    for term in search_terms:
        # Generate mock videos
        videos = generate_mock_videos(term, count)
        results[term] = videos
        
        # Add note about simulated results
        transformations[term] = {
            "note": "Using simulated results. TikTok API restrictions prevent real search results."
        }
    
    return {
        "results": results,
        "transformations": transformations,
        "video_count": sum(len(videos) for videos in results.values()),
        "api_status": "limited_access",
        "notice": "TikTok is currently limiting API access. All results are simulated."
    }

@mcp.tool()
async def get_trending_videos(count: int = 30) -> Dict[str, Any]:
    """Get trending TikTok videos"""
    # Generate diverse trending videos
    trending_topics = [
        "dance challenge", "cooking recipe", "life hack", "workout routine",
        "makeup tutorial", "comedy skit", "pet trick", "travel vlog"
    ]
    
    videos = []
    for i in range(min(count, 15)):
        topic = random.choice(trending_topics)
        username = f"creator{random.randint(1000, 9999)}"
        video_id = f"734{random.randint(1000000, 9999999)}"
        
        videos.append({
            'url': f"https://www.tiktok.com/@{username}/video/{video_id}",
            'description': f"Trending {topic} that everyone is talking about #trending #viral #fyp",
            'author': f"TikTok Creator {random.randint(1, 99)}",
            'stats': {
                'views': str(random.randint(1000000, 10000000)),
                'likes': str(random.randint(100000, 1000000)),
                'shares': str(random.randint(10000, 100000)),
                'comments': str(random.randint(5000, 50000))
            }
        })
    
    return {
        "videos": videos,
        "api_status": "limited_access",
        "notice": "TikTok is currently limiting API access. All results are simulated."
    }

if __name__ == "__main__":
    # Start the MCP server with stdio transport
    logger.info("Starting TikTok Mock Service")
    mcp.run(transport='stdio') 