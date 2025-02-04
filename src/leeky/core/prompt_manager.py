"""Prompt template management functionality."""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path

from .types import PromptTemplate, PromptError

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompt templates and their metadata."""
    
    def __init__(self):
        """Initialize the prompt manager."""
        self._templates: Dict[str, PromptTemplate] = {}
        self._performance_metrics: Dict[str, List[Dict]] = {}

    def register_template(
        self,
        name: str,
        template: str,
        parameters: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        version: str = "1.0.0"
    ) -> PromptTemplate:
        """Register a new prompt template.
        
        Args:
            name: Unique name for the template.
            template: The prompt template string.
            parameters: Dictionary of parameters the template accepts.
            metadata: Additional metadata for the template.
            version: Version string for the template.
            
        Returns:
            The registered PromptTemplate object.
            
        Raises:
            PromptError: If template with same name already exists or template is invalid.
        """
        if name in self._templates:
            raise PromptError(f"Template '{name}' already exists")
            
        template_obj = PromptTemplate(
            template=template,
            name=name,
            version=version,
            parameters=parameters or {},
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        self._templates[name] = template_obj
        self._performance_metrics[name] = []
        
        return template_obj

    def get_template(self, name: str) -> PromptTemplate:
        """Get a prompt template by name.
        
        Args:
            name: Name of the template to retrieve.
            
        Returns:
            The requested PromptTemplate object.
            
        Raises:
            PromptError: If template does not exist.
        """
        if name not in self._templates:
            raise PromptError(f"Template '{name}' not found")
        return self._templates[name]

    def get_all_templates(self) -> List[PromptTemplate]:
        """Get all registered prompt templates.
        
        Returns:
            List of all PromptTemplate objects.
        """
        return list(self._templates.values())

    def update_template(
        self,
        name: str,
        template: Optional[str] = None,
        parameters: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        version: Optional[str] = None
    ) -> PromptTemplate:
        """Update an existing prompt template.
        
        Args:
            name: Name of template to update.
            template: New template string (if updating).
            parameters: New parameters dictionary (if updating).
            metadata: New metadata dictionary (if updating).
            version: New version string (if updating).
            
        Returns:
            The updated PromptTemplate object.
            
        Raises:
            PromptError: If template does not exist.
        """
        if name not in self._templates:
            raise PromptError(f"Template '{name}' not found")
            
        current = self._templates[name]
        
        updated = PromptTemplate(
            template=template or current.template,
            name=name,
            version=version or current.version,
            parameters=parameters or current.parameters.copy(),
            metadata={**current.metadata, **(metadata or {})},
            created_at=current.created_at
        )
        
        self._templates[name] = updated
        return updated

    def delete_template(self, name: str) -> None:
        """Delete a prompt template.
        
        Args:
            name: Name of template to delete.
            
        Raises:
            PromptError: If template does not exist.
        """
        if name not in self._templates:
            raise PromptError(f"Template '{name}' not found")
            
        del self._templates[name]
        del self._performance_metrics[name]

    def track_performance(
        self,
        template_name: str,
        score: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """Track performance metrics for a template.
        
        Args:
            template_name: Name of the template.
            score: Performance score (0.0 to 1.0).
            metadata: Additional metadata about the performance.
            
        Raises:
            PromptError: If template does not exist.
        """
        if template_name not in self._templates:
            raise PromptError(f"Template '{template_name}' not found")
            
        self._performance_metrics[template_name].append({
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })

    def get_performance_metrics(
        self,
        template_name: str
    ) -> List[Dict]:
        """Get performance metrics for a template.
        
        Args:
            template_name: Name of the template.
            
        Returns:
            List of performance metric dictionaries.
            
        Raises:
            PromptError: If template does not exist.
        """
        if template_name not in self._templates:
            raise PromptError(f"Template '{template_name}' not found")
            
        return self._performance_metrics[template_name]

    def save_templates(self, file_path: Union[str, Path]) -> None:
        """Save all templates to a JSON file.
        
        Args:
            file_path: Path to save the templates to.
            
        Raises:
            PromptError: If templates cannot be saved.
        """
        try:
            data = {
                name: {
                    "template": t.template,
                    "version": t.version,
                    "parameters": t.parameters,
                    "metadata": t.metadata,
                    "created_at": t.created_at.isoformat(),
                    "performance": self._performance_metrics[name]
                }
                for name, t in self._templates.items()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise PromptError(f"Failed to save templates: {str(e)}")

    def load_templates(self, file_path: Union[str, Path]) -> None:
        """Load templates from a JSON file.
        
        Args:
            file_path: Path to load the templates from.
            
        Raises:
            PromptError: If templates cannot be loaded.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self._templates.clear()
            self._performance_metrics.clear()
            
            for name, t_data in data.items():
                template = PromptTemplate(
                    template=t_data["template"],
                    name=name,
                    version=t_data["version"],
                    parameters=t_data["parameters"],
                    metadata=t_data["metadata"],
                    created_at=datetime.fromisoformat(t_data["created_at"])
                )
                self._templates[name] = template
                self._performance_metrics[name] = t_data.get("performance", [])
        except Exception as e:
            raise PromptError(f"Failed to load templates: {str(e)}")
