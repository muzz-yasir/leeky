"""
Article scraping pipeline for extracting content from The Intercept articles.
Uses CSV input for URLs and trafilatura for content extraction.
"""

from typing import List, Dict
from pathlib import Path
import json
import time
import requests

import trafilatura
import pandas as pd
from tqdm import tqdm


class ArticleScraper:
    """Scrapes article content using trafilatura library."""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay  # Delay between requests to be respectful
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _clean_article_text(self, text: str) -> str:
        """Remove unwanted sections from article text."""
        if text is None:
            return None
            
        # Split on "WAIT! BEFORE YOU GO" and take only the content before it
        parts = text.split("WAIT! BEFORE YOU GO")
        return parts[0].strip()
        
    def scrape_article(self, url: str) -> Dict[str, str]:
        """Scrape article content from URL."""
        try:
            # Add delay to be respectful to the server
            time.sleep(self.delay)
            
            # Download the webpage
            downloaded = trafilatura.fetch_url(url)
            
            if downloaded is None:
                raise Exception("Failed to download the webpage")
            
            # Extract the main content
            article_text = trafilatura.extract(downloaded, include_comments=False, 
                                             include_tables=False, include_images=False,
                                             output_format='txt')
            
            if article_text is None:
                raise Exception("Failed to extract article content")
                
            # Clean the article text
            article_text = self._clean_article_text(article_text)
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            
            return {
                'url': url,
                'title': metadata.title if metadata else None,
                'text': article_text,
                'author': metadata.author if metadata else None,
                'date': metadata.date if metadata else None,
                'status': 'success'
            }
        except Exception as e:
            return {
                'url': url,
                'status': 'error',
                'error_message': str(e)
            }

class Pipeline:
    """Pipeline for article scraping from CSV input."""
    
    def __init__(self, 
                 scraper: ArticleScraper,
                 output_dir: str = "output"):
        self.scraper = scraper
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_results(self, results: List[Dict[str, str]], 
                    filename: str = "articles.json"):
        """Save results to JSON and CSV files."""
        # Save complete results to JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Create DataFrame with successful scrapes and save to CSV
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            df = pd.DataFrame(successful)
            csv_path = self.output_dir / f"{filename}.csv"
            df.to_csv(csv_path, index=False)

def main():
    # Initialize components
    scraper = ArticleScraper(delay=0.5)
    pipeline = Pipeline(scraper)
    
    # Read URLs from CSV
    df = pd.read_csv('intercept_urls.csv')
    urls_to_scrape = df['URL'].tolist()
    print(f'Loaded {len(urls_to_scrape)} URLs from CSV')
    
    # Scrape articles
    results = []
    for url in tqdm(urls_to_scrape, desc="Scraping articles"):
        result = scraper.scrape_article(url)
        results.append(result)

    pipeline.save_results(results, "intercept_articles")

if __name__ == "__main__":
    main()
