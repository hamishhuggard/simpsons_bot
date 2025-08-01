import json
import csv
from pathlib import Path
from datetime import datetime


class SimpsonsScraperPipeline:
    def __init__(self):
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize CSV file
        self.csv_file = self.data_dir / 'simpsons_episodes.csv'
        self.csv_initialized = False
        
        # Initialize JSON file
        self.json_file = self.data_dir / 'simpsons_episodes.json'
        self.episodes_list = []
        
        # Load existing data if files exist
        self.load_existing_data()

    def load_existing_data(self):
        """Load existing data from files if they exist"""
        if self.json_file.exists():
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    self.episodes_list = json.load(f)
            except json.JSONDecodeError:
                self.episodes_list = []
        
        # Check if CSV headers exist
        if self.csv_file.exists():
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    headers = next(reader)
                    if headers == ['season', 'episode_number_in_season', 'episode_title', 
                                 'episode_url', 'air_date', 'description', 'imdb_rating', 'vote_count']:
                        self.csv_initialized = True
                except StopIteration:
                    pass

    def process_item(self, item, spider):
        """Process each scraped item"""
        # Add timestamp
        item['scraped_at'] = datetime.now().isoformat()
        
        # Add to list for JSON
        self.episodes_list.append(dict(item))
        
        # Save to CSV incrementally
        self.save_to_csv(item)
        
        # Save to JSON periodically (every 10 episodes)
        if len(self.episodes_list) % 10 == 0:
            self.save_to_json()
        
        return item

    def save_to_csv(self, item):
        """Save item to CSV file"""
        if not self.csv_initialized:
            # Write headers
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['season', 'episode_number_in_season', 'episode_title', 
                               'episode_url', 'air_date', 'description', 'imdb_rating', 'vote_count'])
            self.csv_initialized = True
        
        # Append data
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                item['season'],
                item['episode_number_in_season'],
                item['episode_title'],
                item['episode_url'],
                item['air_date'],
                item['description'],
                item['imdb_rating'],
                item['vote_count']
            ])

    def save_to_json(self):
        """Save episodes list to JSON file"""
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.episodes_list, f, ensure_ascii=False, indent=2)

    def close_spider(self, spider):
        """Called when spider is closed"""
        # Final save to JSON
        self.save_to_json()
        
        # Create summary
        summary = {
            'total_episodes': len(self.episodes_list),
            'scraped_at': datetime.now().isoformat(),
            'status': 'completed',
            'files_created': [
                str(self.csv_file),
                str(self.json_file)
            ]
        }
        
        summary_file = self.data_dir / 'scraping_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2) 