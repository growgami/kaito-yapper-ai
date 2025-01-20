import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    # Prompt templates
    BIO_SYSTEM_PROMPT = """You are analyzing Twitter bios to extract professional roles and key information.
Your task is to create a single-line summary for each user that captures their most important roles and achievements.

Guidelines:
- Focus on professional roles, projects, and affiliations
- Include company names with @ symbols
- Highlight key achievements or expertise areas
- Keep each summary to a single line
- Use active voice and present tense
- Separate multiple roles with commas"""

    BIO_USER_PROMPT = """Create a professional summary for each Twitter bio below.
Format: @username: [single line summary]
Example: @example: CEO at @Company, Blockchain Developer, Building web3 infrastructure

Bios to analyze:
{bios}"""

    TWEET_SYSTEM_PROMPT = """You are analyzing tweets to create impactful summaries in Sandra's style (@sandraaleow).
Your task is to identify and summarize the 3 most significant tweets.

Guidelines:
- Select ONLY the 3 most impactful tweets based on content and engagement
- Focus on concrete actions and their impact
- Highlight partnerships, launches, and major announcements
- Keep bullet points concise and factual
- Present each point as a standalone statement
- End with the URL of the single most impactful tweet
- Keep each bullet point to a single line
- Use active voice"""

    TWEET_USER_PROMPT = """Create a concise summary of @{username}'s 3 most impactful tweets, following this format:

{rank}/ @{username} {rank_desc}

Select and summarize ONLY the 3 most significant tweets:
- combine the action and its impact into one line
- focus on major announcements, partnerships, or insights
- end with only the URL of the most impactful tweet

Example format:
- announced major partnership with X to revolutionize Y technology
- launched groundbreaking feature Z with immediate community adoption
- shared detailed analysis of market trends leading to significant discussion

https://x.com/... (URL of the most impactful tweet)

Their tweets:
{tweets}"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 4000, temperature: float = 0.7):
        openai.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache = {}  # In-memory cache

    def cleanup_temp_folders(self):
        """Clean up temporary data folders"""
        folders_to_clean = ['twitter_data', 'processed_data', 'cache']
        for folder in folders_to_clean:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.error(f"Error deleting {file_path}: {e}")
                logger.info(f"Cleaned up {folder} directory")

    def process_tweets(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tweets to extract only essential information"""
        processed_tweets = []
        
        for tweet in tweets:
            # Skip replies
            if tweet.get('isReply', False):
                continue
            
            # Extract only essential fields
            processed_tweet = {
                'url': tweet.get('url', ''),
                'text': tweet.get('text', ''),
                'engagement': {
                    'likes': tweet.get('likeCount', 0),
                    'retweets': tweet.get('retweetCount', 0),
                    'replies': tweet.get('replyCount', 0),
                    'views': tweet.get('viewCount', 0)
                }
            }
            processed_tweets.append(processed_tweet)
        
        return processed_tweets

    async def get_user_descriptions(self, tweets: List[Dict[str, Any]], top_users: List[str]) -> Dict[str, str]:
        """Get user descriptions from their bios"""
        user_bios = {}
        
        # Collect bios for top users
        for tweet in tweets:
            username = tweet.get('author', {}).get('userName')
            if username in top_users and username not in user_bios:
                bio = tweet.get('author', {}).get('profile_bio', {}).get('description', '')
                user_bios[username] = bio

        # Format bios for prompt
        bio_text = ""
        for username, bio in user_bios.items():
            bio_text += f"\n@{username}: {bio}"

        # Get analysis from OpenAI
        bio_response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{
                "role": "system",
                "content": self.BIO_SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": self.BIO_USER_PROMPT.format(bios=bio_text)
            }],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        # Process response into a dictionary
        descriptions = {}
        response_lines = bio_response.choices[0].message['content'].split('\n')
        for line in response_lines:
            if line.startswith('@'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    username = parts[0].strip('@')
                    description = parts[1].strip()
                    descriptions[username] = description

        return descriptions

    async def analyze_user_tweets(self, username: str, tweets: List[Dict[str, Any]], rank: int) -> str:
        """Analyze tweets for a single user"""
        # Format tweets for prompt
        tweet_text = ""
        for tweet in tweets:
            tweet_text += f"\nTweet: {tweet['text']}"
            tweet_text += f"\nEngagement: {tweet['engagement']}"
            tweet_text += f"\nURL: {tweet['url']}\n"

        rank_desc = 'takes top1 yapper today' if rank == 1 else f'is at top {rank}'
        
        response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{
                    "role": "system",
                "content": self.TWEET_SYSTEM_PROMPT
                }, {
                    "role": "user",
                "content": self.TWEET_USER_PROMPT.format(
                    username=username,
                    rank=rank,
                    rank_desc=rank_desc,
                    tweets=tweet_text
                )
                }],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
        return response.choices[0].message['content']

    async def analyze_tweets(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze tweets using OpenAI API"""
        # Get rankings from cache
        rankings = self.cache.get('rankings', [])
        if not rankings:
            raise Exception("No rankings data found in cache")

        # Group tweets by user
        user_tweets: Dict[str, List[Dict[str, Any]]] = {}
        ranked_usernames = [r['username'] for r in rankings]
        
        for tweet in tweets:
            username = tweet.get('author', {}).get('userName', 'unknown')
            if username == 'unknown' or username not in ranked_usernames:
                continue
                
            if username not in user_tweets:
                user_tweets[username] = []
            
            # Process tweet before adding
            processed_tweets = self.process_tweets([tweet])
            if processed_tweets:  # Only add if it's not a reply
                user_tweets[username].extend(processed_tweets)

        # Store processed tweets in cache
        self.cache['processed_tweets'] = user_tweets
        
        # Get user descriptions (1 API call)
        descriptions = await self.get_user_descriptions(tweets, ranked_usernames)
        self.cache['descriptions'] = descriptions
        
        # Analyze each user's tweets (20 API calls)
        analyses = []
        logger.info(f"Processing {len(rankings)} users for trends analysis")
        
        for idx, ranking in enumerate(rankings, 1):
            username = ranking['username']
            logger.info(f"Analyzing tweets for user {idx}/{len(rankings)}: @{username}")
            if username in user_tweets:
                analysis = await self.analyze_user_tweets(username, user_tweets[username], idx)
                analyses.append(analysis)
            logger.info(f"Completed analysis for @{username}")

        self.cache['analyses'] = analyses

        return {
            'descriptions': descriptions,
            'trends_analysis': analyses,
            'rankings': rankings
        }

    async def generate_full_report(self) -> str:
        """Generate a complete analysis report"""
        # Get tweets from cache
        tweets = self.cache.get('tweets', {})
        if not tweets:
            raise Exception("No tweet data found in cache")

        # Combine all tweets into a single list
        all_tweets = []
        for username_tweets in tweets.values():
            all_tweets.extend(username_tweets)
        
        logger.info(f"Loaded {len(all_tweets)} tweets from cache")

        # Analyze tweets
        analysis = await self.analyze_tweets(all_tweets)

        # Ensure reports directory exists
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'reports/analysis_report_{timestamp}.md'

        # Create rankings lookup with score multiplied by 100
        rankings = {
            item['username']: {
                'rank': item['rank'],
                'score': item['score'] * 100  # Multiply by 100 for percentage
            } for item in analysis['rankings']
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Kaito Yapper Analysis Report\n\n")
            
            # Write user descriptions first
            f.write("## Featured Yappers\n\n")
            for username, rank_info in sorted(rankings.items(), key=lambda x: x[1]['rank']):
                description = analysis['descriptions'].get(username, "No description available")
                f.write(f"{rank_info['rank']}. @{username} - {description}\n")

            f.write("\n---\n\n")

            # Write detailed analysis sections
            f.write("## Detailed Analysis\n\n")
            for idx, username in enumerate(sorted(rankings.keys(), key=lambda x: rankings[x]['rank'])):
                rank_info = rankings[username]
                rank_desc = "takes top one" if rank_info['rank'] == 1 else f"ranks at #{rank_info['rank']}"
                f.write(f"### {username} - {rank_desc} | {rank_info['score']:.3f}% of mindshare\n\n")
                
                # Write the analysis for this user
                if idx < len(analysis['trends_analysis']):
                    user_analysis = analysis['trends_analysis'][idx]
                    # Extract and write bullet points
                    lines = user_analysis.split('\n')
                    points = [l for l in lines if l.strip().startswith('-')]
                    
                    if points:
                        for point in points[:3]:  # Only take first 3 points
                            f.write(f"{point}\n")
                        
                        # Extract and write URL
                        urls = [l for l in lines if 'https://' in l]
                        if urls:
                            f.write(f"\n{urls[0]}\n")
                    else:
                        f.write("- No significant tweets in the analyzed timeframe\n")
                else:
                    f.write("- No tweets available for analysis\n")
                
                f.write("\n---\n\n")

        # Clean up after report generation
        self.cleanup_temp_folders()

        logger.info(f"Report generated: {report_file}")
        return report_file 