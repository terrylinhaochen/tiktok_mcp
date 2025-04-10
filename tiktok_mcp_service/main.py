from mcp.server.fastmcp import FastMCP
import logging
from typing import Dict, Any, List, Tuple, Optional
import sys
import os
from dotenv import load_dotenv
import aiohttp

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path modification
try:
    from tiktok_client import TikTokClient
except ImportError:
    # Try alternate import path
    from tiktok_mcp_service.tiktok_client import TikTokClient

# Add this near the top of your main.py file, after imports
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
print(f"Looking for .env file at: {env_path}", file=sys.stderr)
load_dotenv(dotenv_path=env_path)
print(f"ms_token loaded from .env: {'Yes' if os.environ.get('ms_token') else 'No'}", file=sys.stderr)

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from mcp.server import Server
import json
import mcp.server.stdio
from mcp.server.models import InitializationOptions
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TikTok client
tiktok_client = TikTokClient()

@asynccontextmanager
async def lifespan(server: Server) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle."""
    try:
        # Initialize API on startup
        success = await tiktok_client._init_api()
        if success:
            logger.info("TikTok API initialized successfully")
        else:
            logger.warning("TikTok API initialization failed, falling back to HTTP methods")
        
        # Add a delay to ensure the API is fully ready
        await asyncio.sleep(4)
        
        yield {"tiktok_client": tiktok_client}
    finally:
        # Clean up on shutdown
        await tiktok_client.close()
        logger.info("TikTok API shutdown complete")

# Initialize FastMCP app with lifespan
mcp = FastMCP(
    name="TikTok MCP Service",
    description="A Model Context Protocol service for searching TikTok videos",
    version="1.6.0",
    lifespan=lifespan
)

@mcp.resource("status://health")
async def get_health_status() -> Tuple[str, str]:
    """Get the current health status of the service"""
    status = {
        "status": "running",
        "api_initialized": tiktok_client.api is not None,
        "service": {
            "name": "TikTok MCP Service",
            "version": "1.6.0",
            "description": "A Model Context Protocol service for searching TikTok videos"
        }
    }
    return json.dumps(status, indent=2), "application/json"

@mcp.prompt()
def search_prompt(query: str) -> str:
    """Create a prompt for searching TikTok videos"""
    return f"""I'll help you find TikTok videos related to: {query}

IMPORTANT: This service ONLY supports single-word hashtag searches (e.g. #cooking, #snowboarding, #fitness).
Multi-word searches or regular keywords are NOT supported.

Examples of valid searches:
- #cooking
- #recipe 
- #chef
- #snowboard
- #workout

Examples of searches that will NOT work:
- cooking videos
- snowboarding influencer
- professional chef
- workout routine

Would you like me to:
1. Search for videos with specific hashtags (must be single words starting with #)
2. Look for trending videos in this category

Please specify which single-word hashtags you'd like to explore!"""

@mcp.tool()
async def search_videos(search_terms: List[str], count: int = 30) -> str:
    """Search for TikTok videos based on search terms"""
    
    # Limit the count to a reasonable number
    count = min(count, 15)  # Cap at 15 videos max
    
    try:
        all_videos = []
        
        for term in search_terms:
            try:
                # Clean the term - ensure it has a hashtag prefix
                if not term.startswith('#'):
                    term = f"#{term.lstrip('#')}"
                
                logger.info(f"Searching for term: {term}")
                
                # First try HTTP method if API initialization failed
                videos = []
                if not tiktok_client.api:
                    logger.info(f"Using HTTP fallback for search term: {term}")
                    videos = await tiktok_client.search_videos_http(term, count=min(10, count))
                
                # If HTTP method failed or API is initialized, try the API method
                if not videos:
                    videos = await tiktok_client.search_videos(term, count=min(10, count))
                
                # Process the videos
                processed = []
                for video in videos:
                    video_id = video.get('id', '')
                    author = video.get('author', {}).get('uniqueId', '')
                    
                    processed.append({
                        'url': f"https://www.tiktok.com/@{author}/video/{video_id}",
                        'description': video.get('desc', '')[:100] + ('...' if len(video.get('desc', '')) > 100 else ''),
                        'author': video.get('author', {}).get('nickname', ''),
                        'views': str(video.get('stats', {}).get('playCount', 0)),
                        'likes': str(video.get('stats', {}).get('diggCount', 0))
                    })
                
                all_videos.extend(processed)
                logger.info(f"Found {len(processed)} videos for term '{term}'")
                
            except Exception as e:
                logger.error(f"Error searching for term '{term}': {str(e)}")
                continue
        
        # Take only the requested number of videos
        all_videos = all_videos[:count]
        
        # Format the results as JSON string
        return json.dumps(all_videos, indent=2)
        
    except Exception as e:
        error_msg = f"Error in search_videos: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
async def get_trending_videos(count: int = 30) -> Dict[str, Any]:
    """Get trending TikTok videos"""
    logs = []
    errors = {}
    
    # Create a custom log handler to capture logs
    class LogCapture(logging.Handler):
        def emit(self, record):
            logs.append(self.format(record))
    
    # Add our custom handler
    log_capture = LogCapture()
    log_capture.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(log_capture)
    
    try:
        # Ensure API is initialized
        if not tiktok_client.api:
            await tiktok_client._init_api()
            await asyncio.sleep(2)  # Wait for API to be fully ready
            
        videos = await tiktok_client.get_trending_videos(count)
        processed_videos = []
        
        for video in videos:
            processed_videos.append({
                'url': f"https://www.tiktok.com/@{video.get('author', {}).get('uniqueId', '')}/video/{video.get('id')}",
                'description': video.get('desc', ''),
                'stats': {
                    'views': video.get('stats', {}).get('playCount', 0),
                    'likes': video.get('stats', {}).get('diggCount', 0),
                    'shares': video.get('stats', {}).get('shareCount', 0),
                    'comments': video.get('stats', {}).get('commentCount', 0)
                }
            })
        
        logger.info(f"Found {len(processed_videos)} trending videos")
        return {
            "videos": processed_videos,
            "logs": logs,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Error getting trending videos: {str(e)}")
        errors["trending"] = {
            "error": str(e),
            "type": str(type(e).__name__)
        }
        return {
            "videos": [],
            "logs": logs,
            "errors": errors
        }
    finally:
        # Remove our custom handler
        logger.removeHandler(log_capture)

@mcp.tool()
async def search_videos_by_topic(topic: str, count: int = 10) -> Dict[str, Any]:
    """
    Search for TikTok videos related to a specific topic or interest.
    
    Args:
        topic: The topic, interest, or hashtag to search for
        count: Maximum number of videos to return (default: 10)
        
    Returns:
        List of relevant TikTok videos with metadata
    """
    logs = []
    errors = {}
    
    # Add logging capture similar to your get_trending_videos function
    class LogCapture(logging.Handler):
        def emit(self, record):
            logs.append(self.format(record))
    
    log_capture = LogCapture()
    log_capture.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(log_capture)
    
    try:
        # Ensure API is initialized
        if not tiktok_client.api:
            await tiktok_client._init_api()
            await asyncio.sleep(2)
            
        # Format hashtag if needed
        search_term = topic.strip()
        if not search_term.startswith('#') and not ' ' in search_term:
            search_term = f"#{search_term}"
            
        # Search videos by hashtag/topic
        videos = await tiktok_client.search_by_keywords(search_term, count)
        
        # Process results
        processed_videos = []
        for video in videos:
            processed_videos.append({
                'url': f"https://www.tiktok.com/@{video.get('author', {}).get('uniqueId', '')}/video/{video.get('id')}",
                'description': video.get('desc', ''),
                'author': video.get('author', {}).get('uniqueId', ''),
                'stats': {
                    'views': video.get('stats', {}).get('playCount', 0),
                    'likes': video.get('stats', {}).get('diggCount', 0),
                    'shares': video.get('stats', {}).get('shareCount', 0),
                    'comments': video.get('stats', {}).get('commentCount', 0)
                },
                'created_at': video.get('createTime', 0)
            })
            
        logger.info(f"Found {len(processed_videos)} videos for topic '{topic}'")
        return {
            "topic": topic,
            "videos": processed_videos,
            "logs": logs,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Error searching for videos: {str(e)}")
        errors["search"] = {
            "error": str(e),
            "type": str(type(e).__name__)
        }
        return {
            "topic": topic,
            "videos": [],
            "logs": logs,
            "errors": errors
        }
    finally:
        logger.removeHandler(log_capture)

@mcp.tool()
async def analyze_topic_content(videos: List[Dict[str, Any]], openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze content trends and insights from multiple TikTok videos on a topic.
    
    Args:
        videos: List of TikTok video data (from search_videos_by_topic)
        openai_api_key: Optional OpenAI API key for analysis
        
    Returns:
        Analysis of content trends, common themes, and strategic insights
    """
    logs = []
    
    # Add logging capture
    class LogCapture(logging.Handler):
        def emit(self, record):
            logs.append(self.format(record))
    
    log_capture = LogCapture()
    log_capture.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(log_capture)
    
    try:
        # Initialize OpenAI client if API key provided
        if openai_api_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
        else:
            # Use environment variable
            from openai import OpenAI
            client = OpenAI()
            
        # Extract relevant content from videos
        video_data = []
        for idx, video in enumerate(videos[:15]):  # Limit to 15 videos for analysis
            video_data.append({
                "index": idx + 1,
                "description": video.get("description", ""),
                "stats": video.get("stats", {}),
                "url": video.get("url", "")
            })
            
        # Format video data for analysis
        videos_text = "\n".join([
            f"Video {v['index']}: {v['description']} [Views: {v['stats'].get('views', 0)}, " +
            f"Likes: {v['stats'].get('likes', 0)}, Comments: {v['stats'].get('comments', 0)}]"
            for v in video_data
        ])
        
        # Analyze content with OpenAI
        prompt = f"""
        Analyze these TikTok videos on a similar topic:
        
        {videos_text}
        
        Provide:
        1. Common themes and patterns across these videos
        2. Content strategies that seem to be working (based on engagement)
        3. Potential content gaps or opportunities
        4. Recommendations for creating content on this topic
        
        Format your response as JSON:
        {{
            "common_themes": ["theme 1", "theme 2", "theme 3"],
            "successful_strategies": ["strategy 1", "strategy 2"],
            "content_gaps": ["opportunity 1", "opportunity 2"],
            "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a TikTok content strategy expert analyzing video trends."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        # Add aggregate statistics
        total_views = sum(v.get("stats", {}).get("views", 0) for v in videos)
        total_likes = sum(v.get("stats", {}).get("likes", 0) for v in videos)
        total_comments = sum(v.get("stats", {}).get("comments", 0) for v in videos)
        
        engagement_rate = 0
        if total_views > 0:
            engagement_rate = round((total_likes + total_comments) / total_views * 100, 2)
            
        analysis["statistics"] = {
            "total_videos": len(videos),
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "engagement_rate": engagement_rate
        }
        
        # Add top performing videos
        videos_by_engagement = sorted(
            videos, 
            key=lambda v: (v.get("stats", {}).get("likes", 0) + v.get("stats", {}).get("comments", 0)), 
            reverse=True
        )
        
        analysis["top_performing_videos"] = [
            {
                "url": v.get("url"),
                "description": v.get("description"),
                "engagement": v.get("stats", {}).get("likes", 0) + v.get("stats", {}).get("comments", 0)
            }
            for v in videos_by_engagement[:3]  # Top 3 videos
        ]
        
        logger.info(f"Successfully analyzed {len(videos)} videos")
        return {
            "analysis": analysis,
            "logs": logs
        }
        
    except Exception as e:
        logger.error(f"Error analyzing videos: {str(e)}")
        return {
            "error": str(e),
            "logs": logs
        }
    finally:
        logger.removeHandler(log_capture)

@mcp.tool()
async def analyze_hashtag_performance(hashtag: str, count: int = 15) -> Dict[str, Any]:
    """
    Analyze the performance and trends for a specific hashtag.
    
    Args:
        hashtag: The hashtag to analyze (with or without #)
        count: Number of videos to analyze (default: 15)
        
    Returns:
        Analysis of hashtag performance, popularity metrics, and related hashtags
    """
    logs = []
    
    # Add logging capture
    class LogCapture(logging.Handler):
        def emit(self, record):
            logs.append(self.format(record))
    
    log_capture = LogCapture()
    log_capture.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(log_capture)
    
    try:
        # Format hashtag
        clean_hashtag = hashtag.strip()
        if not clean_hashtag.startswith('#'):
            clean_hashtag = f"#{clean_hashtag}"
            
        # Ensure API is initialized
        if not tiktok_client.api:
            await tiktok_client._init_api()
            await asyncio.sleep(2)
            
        # Get hashtag videos
        videos = await tiktok_client.search_by_keywords(clean_hashtag, count)
        
        # Extract all hashtags from video descriptions
        all_hashtags = []
        views_per_day = []
        engagement_rates = []
        
        current_time = time.time()
        
        for video in videos:
            # Extract hashtags from description
            desc = video.get('desc', '')
            tags = [tag.strip() for tag in desc.split('#') if tag.strip()]
            all_hashtags.extend(tags)
            
            # Calculate metrics
            create_time = video.get('createTime', 0)
            age_days = max(1, (current_time - create_time) / 86400)  # Age in days
            
            views = video.get('stats', {}).get('playCount', 0)
            likes = video.get('stats', {}).get('diggCount', 0)
            comments = video.get('stats', {}).get('commentCount', 0)
            
            views_per_day.append(views / age_days)
            
            if views > 0:
                engagement = (likes + comments) / views * 100
                engagement_rates.append(engagement)
                
        # Calculate hashtag metrics
        from collections import Counter
        related_hashtags = Counter(all_hashtags).most_common(10)
        
        avg_views_per_day = sum(views_per_day) / len(views_per_day) if views_per_day else 0
        avg_engagement = sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0
        
        # Return analysis
        return {
            "hashtag": clean_hashtag,
            "videos_analyzed": len(videos),
            "performance": {
                "avg_views_per_day": round(avg_views_per_day, 2),
                "avg_engagement_rate": round(avg_engagement, 2),
                "total_views": sum(v.get('stats', {}).get('playCount', 0) for v in videos)
            },
            "related_hashtags": [
                {"hashtag": f"#{tag}", "count": count}
                for tag, count in related_hashtags
                if tag.lower() != clean_hashtag.lower().replace('#', '')
            ],
            "logs": logs
        }
        
    except Exception as e:
        logger.error(f"Error analyzing hashtag: {str(e)}")
        return {
            "hashtag": clean_hashtag if 'clean_hashtag' in locals() else hashtag,
            "error": str(e),
            "logs": logs
        }
    finally:
        logger.removeHandler(log_capture)

@mcp.tool()
async def create_content_strategy(topic: str, competitor_count: int = 10, openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive TikTok content strategy for a topic including competitive analysis.
    
    Args:
        topic: The topic or niche to create a strategy for
        competitor_count: Number of competitor videos to analyze
        openai_api_key: Optional OpenAI API key for analysis
        
    Returns:
        Detailed content strategy with competitive analysis and recommendations
    """
    logs = []
    
    # Add logging capture
    class LogCapture(logging.Handler):
        def emit(self, record):
            logs.append(self.format(record))
    
    log_capture = LogCapture()
    log_capture.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(log_capture)
    
    try:
        # Step 1: Search for videos
        logger.info(f"Searching for videos on topic: {topic}")
        videos_result = await search_videos_by_topic(topic, competitor_count)
        videos = videos_result.get("videos", [])
        
        if not videos:
            return {
                "topic": topic,
                "error": "No videos found for this topic",
                "logs": logs + videos_result.get("logs", [])
            }
            
        # Step 2: Analyze videos
        logger.info(f"Analyzing {len(videos)} videos for content insights")
        analysis_result = await analyze_topic_content(videos, openai_api_key)
        
        if "error" in analysis_result:
            return {
                "topic": topic,
                "error": analysis_result["error"],
                "logs": logs + analysis_result.get("logs", [])
            }
            
        analysis = analysis_result.get("analysis", {})
        
        # Step 3: Extract relevant hashtags from videos
        all_hashtags = []
        for video in videos:
            desc = video.get('description', '')
            tags = [tag.strip() for tag in desc.split('#') if tag.strip()]
            all_hashtags.extend(tags)
            
        from collections import Counter
        top_hashtags = Counter(all_hashtags).most_common(5)
        
        # Step 4: Generate content strategy
        # Initialize OpenAI client if API key provided
        if openai_api_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
        else:
            # Use environment variable
            from openai import OpenAI
            client = OpenAI()
        
        # Format data for strategy generation
        themes = analysis.get("common_themes", [])
        strategies = analysis.get("successful_strategies", [])
        gaps = analysis.get("content_gaps", [])
        
        hashtags_text = ", ".join([f"#{tag}" for tag, _ in top_hashtags])
        
        prompt = f"""
        Create a comprehensive TikTok content strategy for the topic: {topic}
        
        Based on competitor analysis, we've identified:
        - Common themes: {', '.join(themes)}
        - Successful strategies: {', '.join(strategies)}
        - Content gaps: {', '.join(gaps)}
        - Popular hashtags: {hashtags_text}
        
        Please provide:
        1. Content pillars (3-5 main content categories)
        2. Content ideas for each pillar (2-3 per pillar)
        3. Hashtag strategy
        4. Posting schedule recommendation
        5. Key performance metrics to track
        
        Format your response as JSON:
        {{
            "content_pillars": [
                {{
                    "name": "pillar name",
                    "description": "brief description",
                    "content_ideas": ["idea 1", "idea 2", "idea 3"]
                }}
            ],
            "hashtag_strategy": {{
                "primary_hashtags": ["hashtag1", "hashtag2"],
                "secondary_hashtags": ["hashtag3", "hashtag4"],
                "trending_hashtags": ["hashtag5", "hashtag6"]
            }},
            "posting_schedule": "detailed recommendation",
            "performance_metrics": ["metric 1", "metric 2", "metric 3"]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a TikTok content strategy expert creating actionable plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        strategy = json.loads(response.choices[0].message.content)
        
        # Combine everything into a comprehensive strategy
        return {
            "topic": topic,
            "market_analysis": {
                "common_themes": analysis.get("common_themes", []),
                "successful_strategies": analysis.get("successful_strategies", []),
                "content_gaps": analysis.get("content_gaps", []),
                "top_performing_videos": analysis.get("top_performing_videos", [])
            },
            "content_strategy": strategy,
            "performance_data": analysis.get("statistics", {}),
            "logs": logs + analysis_result.get("logs", [])
        }
        
    except Exception as e:
        logger.error(f"Error creating content strategy: {str(e)}")
        return {
            "topic": topic,
            "error": str(e),
            "logs": logs
        }
    finally:
        logger.removeHandler(log_capture)

@mcp.tool()
async def analyze_video_comments(video_url: str, max_comments: int = 200) -> Dict[str, Any]:
    """
    Perform deep analysis of comments on a TikTok video, including clustering and sentiment analysis.
    
    Args:
        video_url: The URL of the TikTok video to analyze
        max_comments: Maximum number of comments to analyze
        
    Returns:
        Analysis of comments, including clusters, themes, and sentiment
    """
    logs = []
    logs.append(f"Analyzing comments for video: {video_url}")
    
    try:
        # Extract video ID from URL
        video_id = None
        if '/video/' in video_url:
            video_id = video_url.split('/video/')[1].split('?')[0]
        
        if not video_id:
            logs.append("Could not extract video ID from URL")
            return {
                "error": "Invalid video URL format. Expected format: https://www.tiktok.com/@username/video/1234567890123456789",
                "logs": logs
            }
        
        logs.append(f"Extracted video ID: {video_id}")
        
        # Fetch comments for the video
        logs.append(f"Fetching comments (max: {max_comments})...")
        comments = await fetch_video_comments(video_id, max_comments)
        
        if not comments:
            logs.append("No comments found for this video")
            return {
                "message": "No comments found for this video",
                "logs": logs
            }
        
        logs.append(f"Successfully fetched {len(comments)} comments")
        
        # Check for required dependencies
        try:
            import numpy as np
            from sklearn.cluster import KMeans
        except ImportError:
            logs.append("Required libraries 'scikit-learn' and 'numpy' not installed")
            return {
                "error": "Required libraries not installed. Please install them with 'pip install scikit-learn numpy'",
                "logs": logs
            }
        
        # Check for OpenAI dependency
        try:
            from openai import OpenAI
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                logs.append("OpenAI API key not found in environment")
                return {
                    "error": "OpenAI API key is required for comment analysis but was not found in environment",
                    "logs": logs
                }
            
            openai_client = OpenAI(api_key=openai_api_key)
        except ImportError:
            logs.append("Required library 'openai' not installed")
            return {
                "error": "Required library 'openai' not installed. Please install it with 'pip install openai'",
                "logs": logs
            }
        
        # Generate embeddings for comments
        comment_texts = [comment['text'] for comment in comments]
        logs.append("Generating embeddings for comments...")
        
        try:
            embeddings_response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=comment_texts
            )
            embeddings = [embedding.embedding for embedding in embeddings_response.data]
            logs.append("Embeddings generated successfully")
        except Exception as e:
            logs.append(f"Error generating embeddings: {str(e)}")
            return {
                "error": f"Failed to generate embeddings: {str(e)}",
                "logs": logs
            }
        
        # Cluster comments
        logs.append("Clustering comments...")
        X = np.array(embeddings)
        n_clusters = min(5, len(comments) // 10 + 1)  # Dynamic cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        clustered_comments = {i: [] for i in range(n_clusters)}
        for idx, cluster in enumerate(clusters):
            clustered_comments[cluster].append(comments[idx])
        
        logs.append(f"Comments clustered into {n_clusters} groups")
        
        # Prepare the cluster summaries for analysis
        cluster_texts = []
        for cluster_id, cluster_comments in clustered_comments.items():
            cluster_sample = [c['text'] for c in cluster_comments[:10]]  # Take up to 10 comments per cluster
            cluster_texts.append(f"Cluster {cluster_id+1} comments: " + " | ".join(cluster_sample))
        
        # Analyze the clusters using OpenAI
        logs.append("Analyzing comment clusters...")
        try:
            analysis_prompt = f"""Analyze these clusters of TikTok comments for a video:
            
{chr(10).join(cluster_texts)}

For each cluster:
1. Summarize the main opinions or themes
2. Determine the general sentiment (positive, negative, neutral, mixed)
3. Identify any specific feedback or reactions that stand out

Then provide an overall summary of what these comments reveal about reactions to this video.
"""
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing social media comments and identifying themes and sentiments."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            cluster_analysis = response.choices[0].message.content
            logs.append("Analysis completed successfully")
            
        except Exception as e:
            logs.append(f"Error performing cluster analysis: {str(e)}")
            cluster_analysis = "Error performing analysis. Please try again later."
        
        # Prepare the result
        result = {
            "video_id": video_id,
            "total_comments_analyzed": len(comments),
            "clusters": n_clusters,
            "cluster_analysis": cluster_analysis,
            "comment_sample": comments[:5],  # Include a sample of comments
            "logs": logs
        }
        
        return result
        
    except Exception as e:
        logs.append(f"Unexpected error: {str(e)}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "logs": logs
        }

async def fetch_video_comments(video_id: str, max_comments: int = 200) -> List[Dict[str, Any]]:
    """Fetch comments from TikTok API"""
    comments = []
    cursor = 0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.tiktok.com/'
    }
    
    try:
        while len(comments) < max_comments:
            url = "https://www.tiktok.com/api/comment/list/"
            params = {
                'aid': '1988',
                'aweme_id': video_id,
                'count': '50',
                'cursor': cursor
            }
            
            try:
                # Use aiohttp for async requests
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status != 200:
                            logger.error(f"Error fetching comments: status {response.status}")
                            if response.status == 429:
                                logger.warning("Rate limited, waiting before retry...")
                                await asyncio.sleep(10)  # Wait longer for rate limits
                                continue
                            break
                            
                        data = await response.json()
                
                if not data.get('comments'):
                    logger.info("No more comments available")
                    break
                
                for comment in data.get('comments', []):
                    comments.append({
                        'text': comment.get('text', ''),
                        'likes': comment.get('digg_count', 0),
                        'timestamp': comment.get('create_time', 0)
                    })
                
                if len(comments) >= max_comments:
                    logger.info(f"Reached maximum comment count: {max_comments}")
                    break
                    
                cursor = data.get('cursor', 0)
                # Rate limiting to avoid getting blocked
                await asyncio.sleep(1)
                
            except aiohttp.ClientError as e:
                logger.error(f"HTTP request error: {e}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                break
            
        # Sort by likes and take top 40%
        if comments:
            comments.sort(key=lambda x: x.get('likes', 0), reverse=True)
            return comments[:int(len(comments) * 0.4) or 1]  # Ensure at least one comment
        return []
            
    except Exception as e:
        logger.error(f"Error fetching comments: {e}")
        return []

@mcp.tool()
async def analyze_comment_clusters(comments: List[str]) -> Dict[str, Any]:
    """
    Analyze a list of comments by clustering and identifying opinions and sentiment.
    
    Args:
        comments: List of comment texts to analyze
        
    Returns:
        Analysis of comment clusters, themes, and sentiment
    """
    logs = []
    
    try:
        # Format comments for analysis
        formatted_comments = [
            {"text": comment, "likes": 0} for comment in comments
        ]
        
        # Check for required dependencies
        try:
            import numpy as np
            from sklearn.cluster import KMeans
        except ImportError:
            logs.append("Required libraries 'scikit-learn' and 'numpy' not installed")
            return {
                "error": "Required libraries not installed. Please install them with 'pip install scikit-learn numpy'",
                "logs": logs
            }
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                logs.append("OpenAI API key not found in environment")
                return {
                    "error": "OpenAI API key is required for comment analysis but was not found in environment",
                    "logs": logs
                }
            
            openai_client = OpenAI(api_key=openai_api_key)
        except ImportError:
            logs.append("Required library 'openai' not installed")
            return {
                "error": "Required library 'openai' not installed. Please install it with 'pip install openai'",
                "logs": logs
            }
        
        # Generate embeddings for comments
        comment_texts = [comment['text'] for comment in formatted_comments]
        logs.append("Generating embeddings for comments...")
        
        try:
            embeddings_response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=comment_texts
            )
            embeddings = [embedding.embedding for embedding in embeddings_response.data]
            logs.append("Embeddings generated successfully")
        except Exception as e:
            logs.append(f"Error generating embeddings: {str(e)}")
            return {
                "error": f"Failed to generate embeddings: {str(e)}",
                "logs": logs
            }
        
        # Cluster comments
        logs.append("Clustering comments...")
        X = np.array(embeddings)
        n_clusters = min(5, len(formatted_comments) // 10 + 1)  # Dynamic cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        clustered_comments = {i: [] for i in range(n_clusters)}
        for idx, cluster in enumerate(clusters):
            clustered_comments[cluster].append(formatted_comments[idx])
        
        logs.append(f"Comments clustered into {n_clusters} groups")
        
        # Similar analysis as in analyze_video_comments
        # ...

        # Return results
        return {
            "clusters": n_clusters,
            "total_comments": len(comments),
            "clusters_details": {i: [c['text'] for c in clustered_comments[i][:3]] for i in range(n_clusters)},
            "logs": logs
        }
    
    except Exception as e:
        logs.append(f"Unexpected error: {str(e)}")
        return {
            "error": f"Failed to analyze comments: {str(e)}",
            "logs": logs
        }

if __name__ == "__main__":
    # Print debugging info
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
    print(f"PYTHONPATH: {sys.path}", file=sys.stderr)
    print(f"Environment variables: {os.environ.get('ms_token', 'Not set')}", file=sys.stderr)
    
    # Start the MCP server with stdio transport
    logger.info("Starting TikTok MCP Service")
    mcp.run(transport='stdio') 