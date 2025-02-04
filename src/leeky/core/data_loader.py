"""Data loading and preprocessing functionality."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator
import json
import csv
from datetime import datetime

from .types import TextSource, DataLoadError

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of text data from various sources."""

    def __init__(self, config: Dict[str, any]):
        """Initialize the data loader.
        
        Args:
            config: Configuration dictionary for the data loader.
        """
        self.batch_size = config.get("batch_size", 100)
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_dir = Path(config.get("cache_dir", ".cache"))
        
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_from_file(self, file_path: Union[str, Path]) -> TextSource:
        """Load text from a single file.
        
        Args:
            file_path: Path to the file to load.
            
        Returns:
            TextSource object containing the loaded text and metadata.
            
        Raises:
            DataLoadError: If the file cannot be loaded.
        """
        path = Path(file_path)
        try:
            with path.open('r', encoding='utf-8') as f:
                content = f.read()
            
            return TextSource(
                content=content,
                source_id=str(path),
                metadata={
                    "filename": path.name,
                    "file_size": path.stat().st_size,
                    "file_type": path.suffix[1:],  # Remove leading dot
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            raise DataLoadError(f"Failed to load file {file_path}: {str(e)}")

    def load_from_directory(
        self,
        dir_path: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True
    ) -> List[TextSource]:
        """Load text from all matching files in a directory.
        
        Args:
            dir_path: Path to the directory to load from.
            pattern: Glob pattern for matching files.
            recursive: Whether to search subdirectories recursively.
            
        Returns:
            List of TextSource objects.
            
        Raises:
            DataLoadError: If the directory cannot be accessed or files cannot be loaded.
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise DataLoadError(f"Directory not found: {dir_path}")

        try:
            glob_pattern = "**/" + pattern if recursive else pattern
            files = list(path.glob(glob_pattern))
            
            sources = []
            for file_path in files:
                if file_path.is_file():
                    try:
                        sources.append(self.load_from_file(file_path))
                    except DataLoadError as e:
                        logger.warning(f"Skipping file {file_path}: {str(e)}")
            
            return sources
        except Exception as e:
            raise DataLoadError(f"Failed to load from directory {dir_path}: {str(e)}")

    def load_from_json(self, file_path: Union[str, Path]) -> List[TextSource]:
        """Load text sources from a JSON file.
        
        The JSON file should contain an array of objects with at least a 'content' field.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            List of TextSource objects.
            
        Raises:
            DataLoadError: If the JSON file cannot be loaded or has invalid format.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise DataLoadError("JSON file must contain an array of objects")
            
            sources = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict) or 'content' not in item:
                    logger.warning(f"Skipping invalid item at index {idx}")
                    continue
                
                sources.append(TextSource(
                    content=item['content'],
                    source_id=f"{file_path}[{idx}]",
                    metadata={
                        "json_index": idx,
                        **{k: v for k, v in item.items() if k != 'content'}
                    },
                    timestamp=datetime.now()
                ))
            
            return sources
        except Exception as e:
            raise DataLoadError(f"Failed to load JSON file {file_path}: {str(e)}")

    def load_batch(
        self,
        sources: List[TextSource],
        batch_size: Optional[int] = None
    ) -> Iterator[List[TextSource]]:
        """Load text sources in batches.
        
        Args:
            sources: List of TextSource objects to batch.
            batch_size: Size of each batch. If None, uses configured batch size.
            
        Yields:
            Lists of TextSource objects of size batch_size (except possibly the last batch).
        """
        size = batch_size or self.batch_size
        for i in range(0, len(sources), size):
            yield sources[i:i + size]

    def preprocess_text(self, text: str) -> str:
        """Preprocess text content.
        
        Override this method to implement custom preprocessing logic.
        
        Args:
            text: Raw text content to preprocess.
            
        Returns:
            Preprocessed text content.
        """
        # Default implementation: basic cleanup
        return text.strip()

    def _get_cache_path(self, source_id: str) -> Path:
        """Get cache file path for a source ID."""
        return self.cache_dir / f"{hash(source_id)}.json"

    def _cache_source(self, source: TextSource) -> None:
        """Cache a text source if caching is enabled."""
        if not self.cache_enabled:
            return
        
        try:
            cache_path = self._get_cache_path(source.source_id)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open('w', encoding='utf-8') as f:
                json.dump({
                    'content': source.content,
                    'source_id': source.source_id,
                    'metadata': source.metadata,
                    'timestamp': source.timestamp.isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Failed to cache source {source.source_id}: {str(e)}")

    def _load_from_cache(self, source_id: str) -> Optional[TextSource]:
        """Try to load a text source from cache."""
        if not self.cache_enabled:
            return None
            
        cache_path = self._get_cache_path(source_id)
        if not cache_path.exists():
            return None
            
        try:
            with cache_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                return TextSource(
                    content=data['content'],
                    source_id=data['source_id'],
                    metadata=data['metadata'],
                    timestamp=datetime.fromisoformat(data['timestamp'])
                )
        except Exception as e:
            logger.warning(f"Failed to load from cache for {source_id}: {str(e)}")
            return None
