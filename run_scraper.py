#!/usr/bin/env python3
"""
Script to run the Simpsons episode scraper using Scrapy.
"""

import sys
import os
from pathlib import Path

# Add the simpsons_scraper directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'simpsons_scraper'))

def main():
    """Main function to run the scraper"""
    try:
        # Import Scrapy after adding to path
        from scrapy.crawler import CrawlerProcess
        from scrapy.utils.project import get_project_settings
        from simpsons_scraper.spiders.episodes import SimpsonsEpisodesSpider
        
        # Get project settings
        settings = get_project_settings()
        
        # Create crawler process
        process = CrawlerProcess(settings)
        
        # Get max seasons from command line argument or use default
        max_seasons = 40
        if len(sys.argv) > 1:
            try:
                max_seasons = int(sys.argv[1])
            except ValueError:
                print(f"Invalid max_seasons argument: {sys.argv[1]}. Using default value of 40.")
        
        print(f"Starting Simpsons episode scraper with max_seasons={max_seasons}")
        print("This will scrape episodes in parallel and save data incrementally.")
        print("Data will be saved to the 'data' directory.")
        
        # Add spider to process
        process.crawl(SimpsonsEpisodesSpider, max_seasons=max_seasons)
        
        # Start the crawling process
        process.start()
        
        print("\nScraping completed!")
        print("Check the 'data' directory for the scraped files:")
        print("- simpsons_episodes.csv")
        print("- simpsons_episodes.json")
        print("- scraping_summary.json")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running scraper: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 