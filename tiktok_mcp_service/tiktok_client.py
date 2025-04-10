from TikTokApi import TikTokApi
import asyncio
import os
from dotenv import load_dotenv
import logging
import json
from typing import Optional, List, Dict, Any
import aiohttp
import backoff
from functools import lru_cache
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NYC Locations for rotation - All within ~2-3 blocks in Financial District
NYC_LOCATIONS = [
    # Base location - 23 Wall Street
    {"latitude": 40.7075, "longitude": -74.0021, "accuracy": 20, "name": "Wall & Broad"},
    # Around the corner - Federal Hall
    {"latitude": 40.7073, "longitude": -74.0102, "accuracy": 20, "name": "Nassau Street"},
    # Down the block - Near NYSE
    {"latitude": 40.7069, "longitude": -74.0113, "accuracy": 20, "name": "NYSE Area"},
    # Slight variation - Near Chase Plaza
    {"latitude": 40.7077, "longitude": -74.0107, "accuracy": 20, "name": "Chase Plaza"},
    # Small movement - Near Trinity Church
    {"latitude": 40.7081, "longitude": -74.0119, "accuracy": 20, "name": "Trinity Church"}
]

# Track last used location for realistic movement
_last_location_index = 0

# Browser configurations for rotation
BROWSER_CONFIGS = [
    {
        "browser": "firefox",
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
        "viewport": {"width": 1920, "height": 1080}
    },
    {
        "browser": "webkit",
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15",
        "viewport": {"width": 2560, "height": 1440}
    },
    {
        "browser": "chromium",
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "viewport": {"width": 1680, "height": 1050}
    }
]

class TikTokClient:
    def __init__(self):
        """Initialize the TikTok client."""
        self.api = None
        self.session = None
        self.browser = None
        
        # Get environment variables
        self.ms_token = os.environ.get('ms_token')
        self.proxy = os.environ.get('TIKTOK_PROXY')
        
        if not self.ms_token:
            logger.warning("ms_token not found in environment variables. TikTok API functionality will be limited.")
    
    async def _init_api(self, max_retries=3):
        """Initialize the TikTokApi instance with retry mechanism."""
        try:
            # Import here to avoid startup issues if TikTokApi isn't installed
            from TikTokApi import TikTokApi
            
            logger.info("Creating new TikTokApi instance...")
            self.api = TikTokApi()
            logger.info("TikTokApi instance created successfully")
            
            # Initialize session with ms_token
            logger.info("Creating TikTok session...")
            
            # Skip version check since pkg_resources is not available
            logger.info("TikTokApi version check skipped (pkg_resources not available)")
            
            # Use the most compatible parameter set
            try:
                if self.ms_token:
                    self.session = await self.api.create_sessions(
                        ms_tokens=[self.ms_token],
                        num_sessions=1,
                        headless=True
                    )
                else:
                    self.session = await self.api.create_sessions(
                        num_sessions=1,
                        headless=True
                    )
                
                logger.info("TikTok session created successfully")
            except TypeError as e:
                # If there's a TypeError (like unexpected argument), try with fewer parameters
                logger.warning(f"Error with create_sessions parameters: {e}")
                logger.info("Falling back to simpler create_sessions call")
                
                self.session = await self.api.create_sessions(
                    num_sessions=1,
                    headless=True
                )
                logger.info("TikTok session created with fallback parameters")

            # Apply stealth techniques
            logger.info("Applying stealth techniques to session...")
            try:
                from playwright_stealth import stealth_async
                if self.session:
                    for session in self.session:
                        self.browser = session.browser
                        if self.browser:
                            for context in self.browser.contexts:
                                for page in context.pages:
                                    await stealth_async(page)
                    logger.info("Session initialization complete")
                else:
                    logger.warning("No session object available to apply stealth techniques")
            except Exception as stealth_error:
                logger.warning(f"Could not apply stealth techniques: {stealth_error}")
            
            return self.api
            
        except Exception as e:
            logger.error(f"Error initializing TikTok API: {e}")
            logger.error(f"Error type: {type(e)}")
            
            if max_retries > 0:
                logger.info(f"Retrying initialization ({max_retries} attempts left)...")
                await asyncio.sleep(1)  # Wait between retries
                return await self._init_api(max_retries=max_retries-1)
            else:
                logger.error("Maximum retries reached. Failed to initialize TikTok API.")
                return None
    
    async def close(self):
        """Close the TikTok session and browser."""
        try:
            if self.browser:
                await self.browser.close()
                logger.info("Browser closed successfully")
            
            if self.session:
                for session in self.session:
                    await session.close()
                logger.info("TikTok session closed successfully")
                
        except Exception as e:
            logger.error(f"Error closing TikTok session: {e}")
    
    async def search_videos(self, search_term: str, count: int = 30) -> List[Dict[str, Any]]:
        """
        Search for TikTok videos by keyword or hashtag.
        Handles both search_by_keywords and search_for_hashtag based on input.
        """
        try:
            # Initialize API if needed
            if not self.api:
                await self._init_api()
                if not self.api:
                    raise RuntimeError("Failed to initialize TikTok API")
            
            # Determine if this is a hashtag search
            is_hashtag = search_term.startswith('#')
            clean_term = search_term.lstrip('#')
            
            logger.info(f"Searching for {'hashtag' if is_hashtag else 'keyword'}: {clean_term}")
            
            if is_hashtag:
                # Get hashtag info first
                logger.info(f"Getting hashtag info for #{clean_term}...")
                try:
                    hashtag_info = await self.api.hashtag(name=clean_term).info()
                    logger.info(f"Hashtag info received: {json.dumps(hashtag_info, indent=2)}")
                    
                    # Wait a moment before fetching videos to avoid rate limiting
                    logger.info("Waiting before fetching videos...")
                    await asyncio.sleep(1.5)
                    
                    # Now fetch videos for this hashtag
                    logger.info(f"Fetching videos for hashtag #{clean_term}...")
                    hashtag_id = hashtag_info.get('challengeInfo', {}).get('challenge', {}).get('id')
                    
                    if not hashtag_id:
                        logger.error("Could not find hashtag ID in the response")
                        return []
                    
                    logger.info(f"Got hashtag ID: {hashtag_id}")
                    
                    # Fetch videos with the hashtag
                    videos = []
                    challenge = self.api.challenge(id=hashtag_id)
                    
                    # Get videos and convert to dict
                    async for video in challenge.videos(count=count):
                        videos.append(video.as_dict)
                    
                    logger.info(f"Successfully retrieved {len(videos)} videos for hashtag #{clean_term}")
                    return videos
                    
                except Exception as e:
                    logger.error(f"Error in hashtag search: {str(e)}")
                    return []
            
            else:
                # Regular keyword search
                videos = []
                logger.info(f"Searching for keyword: {clean_term}")
                
                # Use the search API
                search_obj = self.api.search.videos(clean_term, count=count)
                async for video in search_obj:
                    videos.append(video.as_dict)
                
                logger.info(f"Successfully retrieved {len(videos)} videos for keyword search: {clean_term}")
                return videos
                
        except Exception as e:
            logger.error(f"Error searching videos: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            return []
    
    async def get_trending(self, count: int = 30) -> List[Dict[str, Any]]:
        """Get trending TikTok videos."""
        try:
            # Initialize API if needed
            if not self.api:
                await self._init_api()
                if not self.api:
                    raise RuntimeError("Failed to initialize TikTok API")
            
            logger.info(f"Fetching {count} trending videos...")
            
            # Get trending videos
            videos = []
            async for video in self.api.trending.videos(count=count):
                videos.append(video.as_dict)
            
            logger.info(f"Successfully retrieved {len(videos)} trending videos")
            return videos

        except Exception as e:
            logger.error(f"Error fetching trending videos: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            return []
    
    async def get_hashtag_info(self, hashtag: str) -> Dict[str, Any]:
        """Get information about a specific hashtag."""
        try:
            # Initialize API if needed
            if not self.api:
                await self._init_api()
                if not self.api:
                    raise RuntimeError("Failed to initialize TikTok API")
            
            # Format hashtag if needed
            clean_hashtag = hashtag.strip()
            if clean_hashtag.startswith('#'):
                clean_hashtag = clean_hashtag[1:]
            
            logger.info(f"Getting hashtag info for: #{clean_hashtag}")
            
            hashtag_info = await self.api.hashtag(name=clean_hashtag).info()
            return hashtag_info
            
        except Exception as e:
            logger.error(f"Error getting hashtag info: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            return {}

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=3,
        max_time=30
    )
    async def _make_request(self, func, *args, **kwargs):
        """Make an API request with retry logic"""
        if not self.api:
            await self._init_api()
            if not self.api:
                raise RuntimeError("Failed to initialize TikTok API")
        
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API request failed: {e}")
            # Force reinitialization on next request
            self.last_init_time = 0
            raise

    @lru_cache(maxsize=100)
    async def get_trending_videos(self, count: int = 30) -> List[Dict[str, Any]]:
        """Get trending videos with caching"""
        videos = []
        try:
            # Initialize API if needed
            if not self.api:
                await self._init_api()
                if not self.api:
                    raise RuntimeError("Failed to initialize TikTok API")
            
            # Get trending videos
            async for video in self.api.trending.videos(count=count):
                videos.append(video.as_dict)
                
        except Exception as e:
            logger.error(f"Failed to get trending videos: {e}")
            raise
        return videos

    async def search_by_keywords(self, keywords, count=30):
        """Search TikTok for videos by keywords or hashtags"""
        logger.info(f"Searching by keywords: {keywords}")
        
        try:
            # Initialize API if needed
            if not self.api:
                await self._init_api()
                if not self.api:
                    raise RuntimeError("Failed to initialize TikTok API")
            
            # Format hashtag if needed
            search_term = keywords.strip()
            if search_term.startswith('#'):
                tag = search_term[1:]  # Remove the # for API call
                logger.info(f"Searching for hashtag: #{tag}")
                
                try:
                    # First try hashtag search
                    videos = []
                    
                    # Get hashtag info
                    hashtag = await self.api.hashtag(name=tag).info()
                    logger.info(f"Found hashtag info with id: {hashtag.get('challengeInfo', {}).get('challenge', {}).get('id')}")
                    
                    # Get videos for the hashtag
                    hashtag_videos = self.api.hashtag(id=hashtag.get('challengeInfo', {}).get('challenge', {}).get('id'))
                    
                    # Collect videos
                    async for video in hashtag_videos.videos(count=count):
                        videos.append(video.as_dict)
                        if len(videos) >= count:
                            break
                    
                    logger.info(f"Found {len(videos)} videos for hashtag #{tag}")
                    return videos
                    
                except Exception as e:
                    logger.error(f"Error in hashtag search: {str(e)}")
                    # Fall back to keyword search
                    logger.info(f"Falling back to keyword search for: {search_term}")
            
            # Use keyword search
            logger.info(f"Performing keyword search for: {search_term}")
            videos = []
            
            async for video in self.api.search.videos(search_term, count=count):
                videos.append(video.as_dict)
                if len(videos) >= count:
                    break
                
            logger.info(f"Found {len(videos)} videos for keyword search: {search_term}")
            return videos
            
        except Exception as e:
            logger.error(f"Search by keywords failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            return []  # Return empty list instead of raising to make function more robust

    async def _ensure_api_initialized(self):
        """Ensure API is initialized"""
        if not self.api:
            await self._init_api()
        await asyncio.sleep(2)  # Wait for API to be fully ready

    async def _handle_api_error(self, func_name, e):
        # Implement exponential backoff logic here
        logger.error(f"Error in {func_name}: {str(e)}")
        # Placeholder for exponential backoff logic
        await asyncio.sleep(1)  # Wait between retries
        return await self._init_api()

    # Alternative approach using direct HTTP requests
    async def search_videos_http(self, search_term, count=30):
        """Search for videos using direct HTTP requests instead of browser automation"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Cookie": f"msToken={self.ms_token}" if self.ms_token else ""
        }
        
        # Use TikTok's search API directly
        async with aiohttp.ClientSession() as session:
            search_url = f"https://www.tiktok.com/api/search/item/?aid=1988&keyword={search_term}&count={count}"
            async with session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("items", [])
                else:
                    return [] 