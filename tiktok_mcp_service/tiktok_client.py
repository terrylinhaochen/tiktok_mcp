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
    
    async def _init_api(self):
        """Initialize the TikTokApi instance."""
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
            
            # Try to create sessions
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
            except Exception as e:
                logger.error(f"Error creating sessions: {str(e)}")
                # Even if session creation fails, we still have the API object
            
            # Return True even if some steps failed - we can still use HTTP fallback
            return True
            
        except Exception as e:
            logger.error(f"Error initializing TikTok API: {str(e)}")
            self.api = None
            return False  # Explicitly return False on failure
    
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
        try:
            # Clean the search term
            clean_term = search_term.strip()
            if clean_term.startswith('#'):
                clean_term = clean_term[1:]
            
            logger.info(f"HTTP Search: Searching for term '{clean_term}' with count {count}")
            
            # TikTok requires these specific headers to work properly
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.tiktok.com/search?q=" + clean_term,  # Important: Include search term in referer
                "Origin": "https://www.tiktok.com",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "Connection": "keep-alive"
            }
            
            # Include cookies if ms_token is available
            if self.ms_token:
                headers["Cookie"] = f"msToken={self.ms_token}; tt_csrf_token=abcd1234; ttwid=random_ttwid_value"
            
            # TikTok requires specific search parameters
            params = {
                "aid": "1988",
                "app_language": "en",
                "app_name": "tiktok_web",
                "battery_info": "1",
                "browser_language": "en-US",
                "browser_name": "Mozilla",
                "browser_online": "true",
                "browser_platform": "MacIntel",
                "browser_version": "5.0 (Macintosh)",
                "channel": "tiktok_web",
                "cookie_enabled": "true",
                "device_id": f"{int(time.time() * 1000)}",
                "device_platform": "web_pc",
                "focus_state": "true",
                "from_page": "search",
                "history_len": "2",
                "is_fullscreen": "false",
                "is_page_visible": "true",
                "keyword": clean_term,
                "count": str(count),
                "cursor": "0",
                "os": "mac",
                "priority_region": "US",
                "referer": f"https://www.tiktok.com/search/video?q={clean_term}",
                "region": "US",
                "screen_height": "1080",
                "screen_width": "1920",
                "type": "1"  # 1 for videos
            }
            
            # Try three different search methods
            search_methods = [
                {
                    "name": "trending method",
                    "url": "https://www.tiktok.com/api/recommend/item_list/",
                    "params": {
                        "aid": "1988", 
                        "count": str(count),
                        "app_language": "en",
                        "device_platform": "web_pc"
                    }
                },
                {
                    "name": "hashtag method",
                    "url": f"https://www.tiktok.com/api/challenge/item_list/",
                    "params": {
                        "aid": "1988",
                        "challengeID": clean_term,
                        "count": str(count)
                    }
                },
                {
                    "name": "search method",
                    "url": "https://www.tiktok.com/api/search/item/",
                    "params": params
                }
            ]
            
            for method in search_methods:
                logger.info(f"Trying {method['name']} at {method['url']}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(method['url'], params=method['params'], headers=headers) as response:
                        logger.info(f"{method['name']} status code: {response.status}")
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            videos = []
                            if "itemList" in data:
                                videos = data["itemList"]
                            elif "items" in data:
                                videos = data["items"]
                            elif "item_list" in data:
                                videos = data["item_list"]
                            
                            logger.info(f"Found {len(videos)} videos with {method['name']}")
                            
                            if videos:
                                # Format the videos
                                formatted_videos = []
                                for video in videos:
                                    formatted_videos.append({
                                        "id": video.get("id", ""),
                                        "desc": video.get("desc", ""),
                                        "author": {
                                            "uniqueId": video.get("author", {}).get("uniqueId", ""),
                                            "nickname": video.get("author", {}).get("nickname", "")
                                        },
                                        "stats": {
                                            "playCount": video.get("stats", {}).get("playCount", 0),
                                            "diggCount": video.get("stats", {}).get("diggCount", 0),
                                            "commentCount": video.get("stats", {}).get("commentCount", 0),
                                            "shareCount": video.get("stats", {}).get("shareCount", 0)
                                        }
                                    })
                                
                                logger.info(f"Successfully formatted {len(formatted_videos)} videos")
                                return formatted_videos
                        
            # If we get here, none of the methods worked
            logger.warning("All HTTP search methods failed")
            
            # As a last resort, try scraping the web page directly
            return await self._scrape_search_results(clean_term, count)
            
        except Exception as e:
            logger.error(f"HTTP Search: Error searching for '{search_term}': {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    async def _scrape_search_results(self, term, count=30):
        """Last resort method - directly scrape search results from the TikTok website"""
        logger.info(f"Using web scraping as last resort for '{term}'")
        
        # URL for direct TikTok search
        url = f"https://www.tiktok.com/search/video?q={term}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        }
        
        try:
            import re
            from bs4 import BeautifulSoup
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Scraping failed with status {response.status}")
                        return []
                    
                    html = await response.text()
                    logger.info(f"Received HTML page of length {len(html)}")
                    
                    # Extract video data from HTML
                    # Look for SIGI_STATE or other JSON data embedded in the page
                    json_data = None
                    
                    # Find SIGI_STATE
                    match = re.search(r'window\[[\'\"]SIGI_STATE[\'\"]\]\s*=\s*(\{.+?\});\s*window\[[\'\"]SIGI_RETRY[\'\"]', html, re.DOTALL)
                    if match:
                        try:
                            json_data = json.loads(match.group(1))
                            logger.info("Found SIGI_STATE data")
                        except json.JSONDecodeError:
                            logger.error("Failed to parse SIGI_STATE JSON")
                    
                    if not json_data:
                        # Try to find ItemModule
                        match = re.search(r'\"ItemModule\"\s*:\s*(\{.+?\}),\s*\"ItemList\"', html, re.DOTALL)
                        if match:
                            try:
                                module_data = "{" + match.group(1) + "}"
                                json_data = {"ItemModule": json.loads(module_data)}
                                logger.info("Found ItemModule data")
                            except json.JSONDecodeError:
                                logger.error("Failed to parse ItemModule JSON")
                    
                    # Extract videos from the JSON data
                    videos = []
                    if json_data and "ItemModule" in json_data:
                        for video_id, video_data in json_data["ItemModule"].items():
                            videos.append({
                                "id": video_id,
                                "desc": video_data.get("desc", ""),
                                "author": {
                                    "uniqueId": video_data.get("author", ""),
                                    "nickname": video_data.get("nickname", "")
                                },
                                "stats": {
                                    "playCount": video_data.get("stats", {}).get("playCount", 0),
                                    "diggCount": video_data.get("stats", {}).get("diggCount", 0),
                                    "commentCount": video_data.get("stats", {}).get("commentCount", 0),
                                    "shareCount": video_data.get("stats", {}).get("shareCount", 0)
                                }
                            })
                            
                            if len(videos) >= count:
                                break
                    
                    logger.info(f"Scraped {len(videos)} videos")
                    return videos
                    
        except Exception as e:
            logger.error(f"Error during web scraping: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    async def search_videos_sigi(self, search_term, count=30):
        """Search for videos using TikTok's SIGI_STATE API - an alternative approach"""
        try:
            # Clean the search term
            clean_term = search_term.strip()
            if clean_term.startswith('#'):
                clean_term = clean_term[1:]
            
            logger.info(f"SIGI Search: Searching for term '{clean_term}'")
            
            # Build the search URL
            url = f"https://www.tiktok.com/tag/{clean_term}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Cookie": f"msToken={self.ms_token}" if self.ms_token else ""
            }
            
            async with aiohttp.ClientSession() as session:
                logger.info(f"Making request to {url}")
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"SIGI Search: Status code {response.status}")
                        return []
                    
                    # Get the HTML content
                    html = await response.text()
                    logger.info(f"Received HTML response of length {len(html)}")
                    
                    # Extract SIGI_STATE JSON
                    try:
                        import re
                        match = re.search(r'window\[\'SIGI_STATE\'\]=(.*?);', html)
                        if not match:
                            logger.error("SIGI_STATE not found in response")
                            return []
                        
                        # Parse the JSON data
                        import json
                        json_data = match.group(1)
                        data = json.loads(json_data)
                        logger.info(f"SIGI_STATE keys: {list(data.keys())}")
                        
                        # Extract videos from the data
                        videos = []
                        if 'ItemModule' in data:
                            for video_id, video_data in data['ItemModule'].items():
                                videos.append({
                                    "id": video_id,
                                    "desc": video_data.get("desc", ""),
                                    "author": {
                                        "uniqueId": video_data.get("author", ""),
                                        "nickname": video_data.get("nickname", "")
                                    },
                                    "stats": {
                                        "playCount": video_data.get("stats", {}).get("playCount", 0),
                                        "diggCount": video_data.get("stats", {}).get("diggCount", 0),
                                        "commentCount": video_data.get("stats", {}).get("commentCount", 0),
                                        "shareCount": video_data.get("stats", {}).get("shareCount", 0)
                                    }
                                })
                                
                                if len(videos) >= count:
                                    break
                                    
                        logger.info(f"SIGI Search: Found {len(videos)} videos")
                        return videos
                        
                    except Exception as e:
                        logger.error(f"Error parsing SIGI_STATE: {str(e)}")
                        return []
                        
        except Exception as e:
            logger.error(f"SIGI Search: Error {str(e)}")
            return [] 

    async def get_mock_search_results(self, term, count=30):
        """Provide mock search results when all other methods fail"""
        logger.warning(f"Using MOCK search results for term '{term}'")
        
        # Clean term for use in descriptions
        clean_term = term.lstrip('#').lower()
        
        # Define topic-specific content based on search term
        topic_content = {
            "tariff": "discussing the economic impact of new tariffs on global trade",
            "trump": "covering Trump's latest campaign events and policy announcements",
            "biden": "analyzing President Biden's recent decisions and policy changes",
            "viral": "that's getting millions of views across TikTok right now",
            "cooking": "showing a delicious recipe that's easy to make at home",
            "workout": "demonstrating effective exercises for strength training",
            "finance": "explaining how to save money and build wealth",
            "news": "breaking down today's top headlines and current events"
        }
        
        # Find the most relevant topic content or use a generic one
        content_description = None
        for keyword, description in topic_content.items():
            if keyword in clean_term:
                content_description = description
                break
        
        if not content_description:
            content_description = f"about {clean_term} that's trending on TikTok"
        
        # Create more realistic usernames related to the topic
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
        
        # Create base videos with more realistic and relevant content
        base_videos = [
            {
                "id": f"734126227922467{random.randint(1000, 9999)}",
                "desc": f"#{clean_term} video {content_description} #viral #trending",
                "author": {
                    "uniqueId": usernames[0],
                    "nickname": usernames[0].replace("_", " ").title()
                },
                "stats": {
                    "playCount": random.randint(800000, 2500000),
                    "diggCount": random.randint(100000, 350000),
                    "commentCount": random.randint(2000, 5000),
                    "shareCount": random.randint(3000, 8000)
                }
            },
            {
                "id": f"734217893612477{random.randint(1000, 9999)}",
                "desc": f"Check out this #{clean_term} content {content_description} #fyp #foryou",
                "author": {
                    "uniqueId": usernames[1],
                    "nickname": usernames[1].replace("_", " ").title()
                },
                "stats": {
                    "playCount": random.randint(1500000, 3000000),
                    "diggCount": random.randint(200000, 450000),
                    "commentCount": random.randint(3000, 6000),
                    "shareCount": random.randint(5000, 10000)
                }
            },
            {
                "id": f"734388739183726{random.randint(1000, 9999)}",
                "desc": f"Latest update on #{clean_term} - {content_description} #tiktok #trending",
                "author": {
                    "uniqueId": usernames[2],
                    "nickname": usernames[2].replace("_", " ").title()
                },
                "stats": {
                    "playCount": random.randint(500000, 1200000),
                    "diggCount": random.randint(80000, 150000),
                    "commentCount": random.randint(1500, 3500),
                    "shareCount": random.randint(2000, 4500)
                }
            }
        ]
        
        # Generate the videos
        videos = []
        for i in range(min(count, 10)):  # Max 10 mock videos
            # Clone one of the base videos and modify it
            idx = i % len(base_videos)
            video = {k: v for k, v in base_videos[idx].items()}
            
            if "author" in video:
                video["author"] = {k: v for k, v in base_videos[idx]["author"].items()}
            
            if "stats" in video:
                video["stats"] = {k: v for k, v in base_videos[idx]["stats"].items()}
            
            # Customize for search term and add variety
            video["id"] = f"{video['id'][:-4]}{random.randint(1000, 9999)}"
            
            # Add variety to stats
            if "stats" in video:
                video["stats"]["playCount"] += random.randint(-200000, 200000)
                video["stats"]["diggCount"] += random.randint(-50000, 50000)
                video["stats"]["commentCount"] += random.randint(-500, 500)
            
            videos.append(video)
        
        return videos 