import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()
ms_token = os.environ.get('ms_token')

async def test_search(term):
    """Test TikTok search directly"""
    print(f"Testing search for: {term}")
    
    # Clean the search term
    clean_term = term.strip()
    if clean_term.startswith('#'):
        clean_term = clean_term[1:]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.tiktok.com/",
        "Cookie": f"msToken={ms_token}" if ms_token else ""
    }
    
    params = {
        "aid": "1988",
        "keyword": clean_term,
        "count": "30",
        "cursor": "0",
        "type": "1"  # 1 for videos
    }
    
    # Try different endpoint variations
    endpoints = [
        "https://www.tiktok.com/api/search/item/",
        "https://www.tiktok.com/api/search/general/",
        "https://www.tiktok.com/api/search/general/full/"
    ]
    
    for endpoint in endpoints:
        print(f"\nTrying endpoint: {endpoint}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params, headers=headers) as response:
                    print(f"Status code: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"Response keys: {list(data.keys())}")
                        
                        # Check different possible response structures
                        videos = data.get("item_list", []) or data.get("data", {}).get("videos", [])
                        print(f"Found {len(videos)} videos")
                        
                        if videos:
                            print(f"First video: {json.dumps(videos[0], indent=2)[:500]}...")
                        else:
                            print(f"No videos found. Response: {json.dumps(data)[:500]}...")
                    else:
                        print(f"Error response: {await response.text()[:200]}...")
        except Exception as e:
            print(f"Error with endpoint {endpoint}: {e}")

async def main():
    # Test with various search terms
    test_terms = [
        "#tariffs",
        "tariffs",
        "#trump",
        "#biden",
        "#viral"  # This should definitely return results
    ]
    
    for term in test_terms:
        await test_search(term)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main()) 