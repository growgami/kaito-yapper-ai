import asyncio
import os
import sys
from datetime import datetime
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import json

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.kaito_system.kaito_leaderboard import KaitoLeaderboard
from scripts.kaito_system.twitter_scraper import TwitterScraper
from scripts.kaito_system.report_generator import ReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/kaito_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_environment() -> Dict[str, Any]:
    """Load and validate environment variables"""
    load_dotenv()
    
    # Validate required environment variables
    required_vars = ['OPENAI_API_KEY', 'APIFY_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'apify_api_key': os.getenv('APIFY_API_KEY'),
        'model_name': os.getenv('MODEL_NAME', 'gpt-4-turbo-preview'),
        'max_tokens': int(os.getenv('MAX_TOKENS', '4000')),
        'temperature': float(os.getenv('TEMPERATURE', '0.7')),
        'kaito_timeframe': os.getenv('KAITO_TIMEFRAME', '7d')
    }

async def run_analysis(config: Dict[str, Any]) -> None:
    """
    Run the complete Kaito analysis pipeline
    """
    try:
        # Step 1: Process Kaito leaderboard
        logger.info("Starting Kaito leaderboard processing...")
        kaito = KaitoLeaderboard(
            timeframe=config.get('kaito_timeframe', '7d')
        )
        
        # Get top 20 accounts
        top_20 = kaito.get_leaderboard()
        logger.info("Kaito data processed and cached")
        
        # Initialize Twitter scraper
        twitter = TwitterScraper(config['apify_api_key'])
        
        # Get usernames from top 20
        usernames = [acc['username'] for acc in top_20]
        
        # Scrape tweets for all users
        logger.info("Starting Twitter data scraping...")
        await twitter.scrape_multiple_users(usernames)
        logger.info("Twitter data scraping completed and cached")

        # Generate analysis report
        logger.info("Starting analysis generation...")
        generator = ReportGenerator(
            api_key=config['openai_api_key'],
            model=config['model_name'],
            max_tokens=config['max_tokens'],
            temperature=config['temperature']
        )
        
        # Pass the cached data to the report generator
        generator.cache['rankings'] = top_20
        generator.cache['tweets'] = twitter.cache
        
        report_file = await generator.generate_full_report()
        logger.info(f"Analysis report generated: {report_file}")

        # Clean up temporary data
        kaito.cache.clear()
        twitter.cache.clear()
        generator.cache.clear()

        logger.info("Complete analysis pipeline finished successfully!")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}", exc_info=True)
        raise

def ensure_reports_directory():
    """Ensure reports directory exists"""
    if not os.path.exists('reports'):
        os.makedirs('reports')
        logger.info("Created reports directory")

def main():
    # Ensure reports directory exists
    ensure_reports_directory()
    
    # Load and validate environment variables
    config = load_environment()
    
    # Run the analysis
    try:
        asyncio.run(run_analysis(config))
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 