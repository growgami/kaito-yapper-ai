import os
import json
from datetime import datetime
import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class KaitoLeaderboard:
    def __init__(self, timeframe: str = "7d"):
        self.base_url = "https://hub.kaito.ai"
        self.timeframe = timeframe
        self.headers = {
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json",
            "origin": "https://yaps.kaito.ai",
            "referer": "https://yaps.kaito.ai/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
        }
        self.cache = {}

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Fetch top 20 accounts from Kaito leaderboard"""
        endpoint = f"/api/v1/gateway/ai"
        params = {
            "duration": self.timeframe,
            "topic_id": "",
            "top_n": 100  # Get full list to ensure we have top 20
        }
        
        payload = {
            "path": "/api/yapper/public_kol_mindshare_leaderboard",
            "method": "GET",
            "params": params,
            "body": {}
        }
        
        try:
            logger.info(f"Making request to {self.base_url}{endpoint}")
            
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json=payload
            )
            
            logger.info(f"Status Code: {response.status_code}")
            
            if response.status_code in [200, 201]:
                data = response.json()
                if not data:
                    raise Exception(f"Empty response")
                
                # Store raw data in cache
                self.cache['raw_data'] = data
                
                # Process and return top 20
                top_20 = []
                for idx, account in enumerate(data[:20], 1):  # Only take top 20
                    if 'username' not in account:
                        continue
                    top_20.append({
                        'username': account['username'],
                        'rank': idx,
                        'score': float(account.get('mindshare', 0)),
                        'followers': int(account.get('follower_count', 0))
                    })
                
                # Store processed data in cache
                self.cache['top_20'] = top_20
                
                logger.info(f"Successfully processed top 20 accounts")
                return top_20
            else:
                logger.error(f"Request failed with status code: {response.status_code}")
                raise Exception(f"Failed to fetch Kaito data: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error making request: {str(e)}")
            raise 