import scrapy
import re
import json
import csv
from pathlib import Path
from datetime import datetime


class SimpsonsEpisodesSpider(scrapy.Spider):
    name = 'simpsons_episodes'
    allowed_domains = ['imdb.com']
    start_urls = ['https://www.imdb.com/title/tt0096697/episodes/']
    
    custom_settings = {
        'CONCURRENT_REQUESTS': 16,  # Parallel requests
        'DOWNLOAD_DELAY': 0.5,  # Reduced delay for faster scraping
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'FEEDS': {
            'simpsons_episodes.json': {
                'format': 'json',
                'encoding': 'utf8',
                'indent': 2,
                'overwrite': False,  # Append to existing file
            },
            'simpsons_episodes.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'overwrite': False,  # Append to existing file
            }
        }
    }
    
    def __init__(self, max_seasons=40, *args, **kwargs):
        super(SimpsonsEpisodesSpider, self).__init__(*args, **kwargs)
        self.max_seasons = int(max_seasons)
        self.episodes_found = 0
        
        # Create data directory if it doesn't exist
        Path('data').mkdir(exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        csv_file = Path('data/simpsons_episodes.csv')
        if not csv_file.exists():
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['season', 'episode_number_in_season', 'episode_title', 
                               'episode_url', 'air_date', 'description', 'imdb_rating', 'vote_count'])

    def parse(self, response):
        """Parse the main episodes page and generate season URLs"""
        self.logger.info(f"Starting to scrape Simpsons episodes (max seasons: {self.max_seasons})")
        
        # Generate URLs for each season
        for season in range(1, self.max_seasons + 1):
            season_url = f"https://www.imdb.com/title/tt0096697/episodes/?season={season}&ref_=ttep"
            yield scrapy.Request(
                season_url,
                callback=self.parse_season,
                meta={'season': season},
                dont_filter=True
            )

    def parse_season(self, response):
        """Parse individual season pages"""
        season = response.meta['season']
        self.logger.info(f"Parsing Season {season}")
        
        # Find all episode title elements
        episode_titles = response.css('h4[data-testid="slate-list-card-title"]')
        
        if not episode_titles:
            self.logger.info(f"No episodes found for Season {season}. Stopping.")
            return
        
        for title_elem in episode_titles:
            episode_data = self.extract_episode_data(title_elem, season)
            if episode_data:
                self.episodes_found += 1
                yield episode_data
                
                # Save incrementally to CSV
                self.save_episode_incrementally(episode_data)

    def extract_episode_data(self, title_elem, season):
        """Extract episode information from the title element"""
        episode_data = {
            'season': season,
            'episode_number_in_season': 'N/A',
            'episode_title': 'N/A',
            'episode_url': 'N/A',
            'air_date': 'N/A',
            'description': 'N/A',
            'imdb_rating': 'N/A',
            'vote_count': 'N/A'
        }
        
        # Get the episode container (parent of parent of h4)
        episode_container = title_elem.xpath('./ancestor::div[2]').get()
        if not episode_container:
            return None
        
        # Extract episode title and URL
        title_link = title_elem.css('a.ipc-title-link-wrapper')
        if title_link:
            title_text_div = title_link.css('div.ipc-title__text--reduced::text').get()
            if title_text_div:
                episode_data['episode_title'] = title_text_div.strip()
            
            href = title_link.attrib.get('href')
            if href:
                episode_data['episode_url'] = f"https://www.imdb.com{href}"
            
            # Extract episode number from full title
            full_title = title_elem.css('div.ipc-title__text::text').get()
            if full_title:
                match = re.search(r'S\d+\.E(\d+)', full_title)
                if match:
                    episode_data['episode_number_in_season'] = int(match.group(1))
        
        # Extract air date
        air_date = title_elem.xpath('./ancestor::div[2]//span[contains(@class, "knzESm")]/text()').get()
        if air_date:
            episode_data['air_date'] = air_date.strip()
        
        # Extract description
        description = title_elem.xpath('./ancestor::div[2]//div[contains(@class, "ipc-html-content-inner-div")]/text()').get()
        if description:
            episode_data['description'] = description.strip()
        
        # Extract rating and vote count
        rating_group = title_elem.xpath('./ancestor::div[2]//span[@data-testid="ratingGroup--imdb-rating"]')
        if rating_group:
            rating = rating_group.css('span.ipc-rating-star--rating::text').get()
            if rating:
                episode_data['imdb_rating'] = rating.strip()
            
            vote_count = rating_group.css('span.ipc-rating-star--voteCount::text').get()
            if vote_count:
                # Clean vote count text
                votes_text = vote_count.strip().replace("(", "").replace(")", "").replace("\u00a0", "")
                episode_data['vote_count'] = votes_text
        
        return episode_data

    def save_episode_incrementally(self, episode_data):
        """Save episode data incrementally to CSV file"""
        csv_file = Path('data/simpsons_episodes.csv')
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode_data['season'],
                episode_data['episode_number_in_season'],
                episode_data['episode_title'],
                episode_data['episode_url'],
                episode_data['air_date'],
                episode_data['description'],
                episode_data['imdb_rating'],
                episode_data['vote_count']
            ])

    def closed(self, reason):
        """Called when spider is closed"""
        self.logger.info(f"Spider closed. Total episodes scraped: {self.episodes_found}")
        
        # Create a summary file
        summary = {
            'total_episodes': self.episodes_found,
            'scraped_at': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        with open('data/scraping_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2) 