"""Article scraping functionality."""

import trafilatura
import re

class ArticleScraper:
    """Scraper for extracting and cleaning article content."""
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing formatting and unwanted content."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # Named entities like &nbsp;
        text = re.sub(r'&#[0-9]+;', ' ', text)    # Numbered entities like &#160;
        
        # Replace common HTML artifacts and Unicode characters
        replacements = {
            '\u2018': "'",  # Smart single quotes
            '\u2019': "'",
            '\u201c': '"',  # Smart double quotes
            '\u201d': '"',
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
            '\xa0': ' ',    # Non-breaking space
            '\t': ' ',      # Tabs to spaces
            '&nbsp;': ' ',  # In case any HTML entities remain
            '&quot;': '"',
            '&apos;': "'",
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Split into lines and clean each line
        lines = []
        for line in text.split('\n'):
            # Clean the line
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip very short lines (likely nav items)
            if len(line) < 20:
                continue
                
            # Skip lines that are likely navigation/menu items
            skip_patterns = ['menu', 'navigation', 'search', 'subscribe', 
                           'sign up', 'follow us', 'share this', 'copyright',
                           'all rights reserved', 'terms of use', 'privacy policy',
                           'advertisement', 'sponsored', 'related articles']
            if any(pattern in line.lower() for pattern in skip_patterns):
                continue
            
            # Remove any remaining HTML-style formatting
            line = re.sub(r'\[.*?\]', '', line)  # Remove square bracket content
            line = re.sub(r'\{.*?\}', '', line)  # Remove curly bracket content
            
            # Normalize whitespace
            line = ' '.join(line.split())
            
            # Clean up punctuation
            line = re.sub(r'\.{2,}', '.', line)  # Multiple periods to single
            line = re.sub(r'-{2,}', '-', line)   # Multiple dashes to single
            line = re.sub(r'\s*([.,!?])', r'\1', line)  # Remove space before punctuation
            line = re.sub(r'([.,!?])\s*([.,!?])', r'\1', line)  # Remove duplicate punctuation
            
            # Final whitespace cleanup
            line = line.strip()
            if line:
                lines.append(line)
        
        # Join lines with single newlines, ensuring no empty lines
        return '\n'.join(line for line in lines if line.strip())

    def scrape_url(self, url: str) -> str:
        """Scrape article content from URL."""
        try:
            # Configure trafilatura settings for news articles/blog posts
            config = {
                'include_comments': False,
                'include_tables': False,
                'include_images': False,
                'include_links': False,
                'no_fallback': False,  # Allow fallback methods if main extraction fails
                'target_language': 'en',
                'deduplicate': True,
                'output_format': 'txt'
            }
            
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                raise Exception("Failed to download the webpage")
            
            # Extract main content with custom config
            article_text = trafilatura.extract(
                downloaded,
                **config
            )
            
            if article_text is None:
                # Try fallback with minimal config
                article_text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    include_images=False,
                    output_format='txt'
                )
                
            if article_text is None:
                raise Exception("Failed to extract article content")
            
            # Clean and process the extracted text
            cleaned_text = self.clean_text(article_text)
            
            if not cleaned_text:
                raise Exception("No relevant content found after cleaning")
            
            return cleaned_text
            
        except Exception as e:
            raise Exception(f"Failed to scrape URL: {str(e)}")
