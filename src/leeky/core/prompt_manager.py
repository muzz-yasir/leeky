"""Prompt template management functionality."""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path

from .types import PromptTemplate, PromptError, TemplateType

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompt templates and their metadata."""
    
    def __init__(self):
        """Initialize the prompt manager."""
        self._instruction_templates: Dict[str, PromptTemplate] = {}
        self._jailbreak_templates: Dict[str, PromptTemplate] = {}
        self._performance_metrics: Dict[str, List[Dict]] = {}

    def register_template(
        self,
        name: str,
        template: str,
        template_type: TemplateType,
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
        templates_dict = (self._instruction_templates if template_type == TemplateType.INSTRUCTION 
                         else self._jailbreak_templates)
        
        if name in templates_dict:
            raise PromptError(f"Template '{name}' already exists")
            
        template_obj = PromptTemplate(
            template=template,
            name=name,
            version=version,
            parameters=parameters or {},
            metadata=metadata or {},
            created_at=datetime.now(),
            template_type=template_type
        )
        
        templates_dict[name] = template_obj
        self._performance_metrics[name] = []
        
        return template_obj

    def get_template(self, name: str, template_type: TemplateType) -> PromptTemplate:
        """Get a prompt template by name.
        
        Args:
            name: Name of the template to retrieve.
            
        Returns:
            The requested PromptTemplate object.
            
        Raises:
            PromptError: If template does not exist.
        """
        templates_dict = (self._instruction_templates if template_type == TemplateType.INSTRUCTION 
                         else self._jailbreak_templates)
        
        if name not in templates_dict:
            raise PromptError(f"Template '{name}' not found")
        return templates_dict[name]

    def get_all_templates(self, template_type: Optional[TemplateType] = None) -> List[PromptTemplate]:
        """Get all registered prompt templates.
        
        Args:
            template_type: Optional filter for template type.
            
        Returns:
            List of all PromptTemplate objects.
        """
        if template_type == TemplateType.INSTRUCTION:
            return list(self._instruction_templates.values())
        elif template_type == TemplateType.JAILBREAK:
            return list(self._jailbreak_templates.values())
        else:
            return (list(self._instruction_templates.values()) + 
                   list(self._jailbreak_templates.values()))

    def update_template(
        self,
        name: str,
        template_type: TemplateType,
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
        templates_dict = (self._instruction_templates if template_type == TemplateType.INSTRUCTION 
                         else self._jailbreak_templates)
        
        if name not in templates_dict:
            raise PromptError(f"Template '{name}' not found")
            
        current = templates_dict[name]
        
        updated = PromptTemplate(
            template=template or current.template,
            name=name,
            version=version or current.version,
            parameters=parameters or current.parameters.copy(),
            metadata={**current.metadata, **(metadata or {})},
            created_at=current.created_at,
            template_type=template_type
        )
        
        templates_dict[name] = updated
        return updated

    def delete_template(self, name: str, template_type: TemplateType) -> None:
        """Delete a prompt template.
        
        Args:
            name: Name of template to delete.
            
        Raises:
            PromptError: If template does not exist.
        """
        templates_dict = (self._instruction_templates if template_type == TemplateType.INSTRUCTION 
                         else self._jailbreak_templates)
        
        if name not in templates_dict:
            raise PromptError(f"Template '{name}' not found")
            
        del templates_dict[name]
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
        # Check both template dictionaries
        template = None
        if template_name in self._instruction_templates:
            template = self._instruction_templates[template_name]
        elif template_name in self._jailbreak_templates:
            template = self._jailbreak_templates[template_name]
            
        if not template:
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
        # Check both template dictionaries
        template = None
        if template_name in self._instruction_templates:
            template = self._instruction_templates[template_name]
        elif template_name in self._jailbreak_templates:
            template = self._jailbreak_templates[template_name]
            
        if not template:
            raise PromptError(f"Template '{template_name}' not found")
            
        return self._performance_metrics[template_name]

    def combine_templates(
        self,
        instruction_name: str,
        jailbreak_name: Optional[str] = None
    ) -> str:
        """Combine instruction and jailbreak templates.
        
        Args:
            instruction_name: Name of the instruction template.
            jailbreak_name: Optional name of the jailbreak template.
            
        Returns:
            Combined template string.
            
        Raises:
            PromptError: If templates not found.
        """
        instruction_template = self.get_template(instruction_name, TemplateType.INSTRUCTION)
        final_template = instruction_template.template
        
        if jailbreak_name:
            jailbreak_template = self.get_template(jailbreak_name, TemplateType.JAILBREAK)
            final_template = f"{jailbreak_template.template}\n\n{final_template}"
            
        return final_template

    def save_templates(self, instruction_path: Union[str, Path], jailbreak_path: Union[str, Path]) -> None:
        """Save all templates to JSON files.
        
        Args:
            instruction_path: Path to save instruction templates to.
            jailbreak_path: Path to save jailbreak templates to.
            
        Raises:
            PromptError: If templates cannot be saved.
        """
        try:
            # Save instruction templates
            instruction_data = {
                name: {
                    "template": t.template,
                    "version": t.version,
                    "parameters": t.parameters,
                    "metadata": t.metadata,
                    "created_at": t.created_at.isoformat(),
                    "performance": self._performance_metrics.get(name, [])
                }
                for name, t in self._instruction_templates.items()
            }
            
            with open(instruction_path, 'w', encoding='utf-8') as f:
                json.dump(instruction_data, f, indent=2)
                
            # Save jailbreak templates
            jailbreak_data = {
                name: {
                    "template": t.template,
                    "version": t.version,
                    "parameters": t.parameters,
                    "metadata": t.metadata,
                    "created_at": t.created_at.isoformat(),
                    "performance": self._performance_metrics.get(name, [])
                }
                for name, t in self._jailbreak_templates.items()
            }
            
            with open(jailbreak_path, 'w', encoding='utf-8') as f:
                json.dump(jailbreak_data, f, indent=2)
        except Exception as e:
            raise PromptError(f"Failed to save templates: {str(e)}")

    def load_templates(self, instruction_path: Union[str, Path], jailbreak_path: Union[str, Path]) -> None:
        """Load templates from JSON files.
        
        Args:
            instruction_path: Path to load instruction templates from.
            jailbreak_path: Path to load jailbreak templates from.
            
        Raises:
            PromptError: If templates cannot be loaded.
        """
        try:
            # Load instruction templates
            with open(instruction_path, 'r', encoding='utf-8') as f:
                instruction_data = json.load(f)
                
            self._instruction_templates.clear()
            
            for name, t_data in instruction_data.items():
                template = PromptTemplate(
                    template=t_data["template"],
                    name=name,
                    version=t_data["version"],
                    parameters=t_data["parameters"],
                    metadata=t_data["metadata"],
                    created_at=datetime.fromisoformat(t_data["created_at"]),
                    template_type=TemplateType.INSTRUCTION
                )
                self._instruction_templates[name] = template
                self._performance_metrics[name] = t_data.get("performance", [])
                
            # Load jailbreak templates
            with open(jailbreak_path, 'r', encoding='utf-8') as f:
                jailbreak_data = json.load(f)
                
            self._jailbreak_templates.clear()
            
            for name, t_data in jailbreak_data.items():
                template = PromptTemplate(
                    template=t_data["template"],
                    name=name,
                    version=t_data["version"],
                    parameters=t_data["parameters"],
                    metadata=t_data["metadata"],
                    created_at=datetime.fromisoformat(t_data["created_at"]),
                    template_type=TemplateType.JAILBREAK
                )
                self._jailbreak_templates[name] = template
                self._performance_metrics[name] = t_data.get("performance", [])
        except Exception as e:
            raise PromptError(f"Failed to load templates: {str(e)}")
