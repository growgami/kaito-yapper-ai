from apify_client import ApifyClient
import json
from datetime import datetime
import os
import asyncio
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TwitterScraper:
    def __init__(self, api_token: str):
        self.client = ApifyClient(api_token)
        self.cache = {}
        
    def scrape_user(self, username: str):
        """Run single user scrape exactly like twitter_scraper.py"""
        run_input = {
            "filter:blue_verified": False,
            "filter:consumer_video": False,
            "filter:has_engagement": False,
            "filter:hashtags": False,
            "filter:images": False,
            "filter:links": False,
            "filter:media": False,
            "filter:mentions": False,
            "filter:native_video": False,
            "filter:nativeretweets": False,
            "filter:news": False,
            "filter:pro_video": False,
            "filter:quote": False,
            "filter:replies": False,
            "filter:safe": False,
            "filter:spaces": False,
            "filter:twimg": False,
            "filter:verified": False,
            "filter:videos": False,
            "filter:vine": False,
            "from": username,  # Just change the username here
            "include:nativeretweets": False,
            "lang": "en",
            "maxItems": 60,
            "queryType": "Top",
            "within_time": "1d"
        }

        try:
            # Run the Actor and wait for it to finish
            logger.info(f"Starting Twitter scrape for {username}...")
            run = self.client.actor("CJdippxWmn9uRfooo").call(run_input=run_input)
            
            # Fetch Actor results
            logger.info(f"Fetching results for {username}...")
            results = []
            dataset = self.client.dataset(run["defaultDatasetId"])
            for item in dataset.iterate_items():
                results.append(item)
            
            # Store results in cache by username
            self.cache[username] = results
            
            logger.info(f"Data cached for {username}")
            logger.info(f"Total tweets collected for {username}: {len(results)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error scraping {username}: {str(e)}")
            return []

    async def scrape_multiple_users(self, usernames: List[str]):
        """Run multiple instances concurrently using asyncio.to_thread"""
        tasks = [asyncio.to_thread(self.scrape_user, username) for username in usernames]
        results = await asyncio.gather(*tasks)
        
        # Results are already cached by username in scrape_user
        return results
        
    def get_cached_data(self, timestamp: str = None) -> Dict[str, Any]:
        """Get cached data for a specific timestamp or latest"""
        if timestamp:
            return {
                k: v for k, v in self.cache.items() 
                if k.endswith(timestamp)
            }
        else:
            # Get latest timestamp
            timestamps = set(k.split('_')[-1] for k in self.cache.keys())
            if not timestamps:
                return {}
            latest = max(timestamps)
            return {
                k: v for k, v in self.cache.items() 
                if k.endswith(latest)
            } 