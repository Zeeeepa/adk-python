# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Template manager for creating, storing, and managing agent templates."""

from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..agents.base_agent import BaseAgent
from ..agents.llm_agent import LlmAgent
from ..agents.loop_agent import LoopAgent
from ..agents.parallel_agent import ParallelAgent
from ..agents.sequential_agent import SequentialAgent
from .models.agent_template import AgentTemplate
from .storage.template_storage import TemplateStorage
from .validation.template_validator import TemplateValidator

logger = logging.getLogger('google_adk.orchestrator.template_manager')


class TemplateManager:
  """Manager for agent templates with CRUD operations and validation.
  
  The TemplateManager provides comprehensive template management including:
  - Template creation, modification, and deletion
  - Template validation and versioning
  - Agent instantiation from templates
  - Template inheritance and composition
  - Template discovery and search
  """
  
  def __init__(
      self,
      storage: Optional[TemplateStorage] = None,
      validator: Optional[TemplateValidator] = None
  ):
    """Initialize the template manager.
    
    Args:
      storage: Template storage backend (defaults to in-memory)
      validator: Template validator (defaults to basic validator)
    """
    self.storage = storage or TemplateStorage()
    self.validator = validator or TemplateValidator()
    self._agent_type_map = {
        "LlmAgent": LlmAgent,
        "SequentialAgent": SequentialAgent,
        "ParallelAgent": ParallelAgent,
        "LoopAgent": LoopAgent,
    }
    
    logger.info("Template manager initialized")
  
  def create_template(self, template: AgentTemplate) -> str:
    """Create a new agent template.
    
    Args:
      template: The template to create
    
    Returns:
      The ID of the created template
    
    Raises:
      ValueError: If template validation fails
    """
    # Validate the template
    validation_result = self.validator.validate_template(template)
    if not validation_result.is_valid:
      raise ValueError(f"Template validation failed: {validation_result.errors}")
    
    # Check for name conflicts
    existing = self.storage.get_template_by_name(template.name, template.version)
    if existing:
      raise ValueError(f"Template {template.name}:{template.version} already exists")
    
    # Store the template
    template_id = self.storage.store_template(template)
    
    logger.info(f"Created template {template.name}:{template.version} with ID {template_id}")
    return template_id
  
  def get_template(self, template_id: str) -> Optional[AgentTemplate]:
    """Get a template by ID.
    
    Args:
      template_id: The template ID
    
    Returns:
      The template if found, None otherwise
    """
    return self.storage.get_template(template_id)
  
  def get_template_by_name(self, name: str, version: Optional[str] = None) -> Optional[AgentTemplate]:
    """Get a template by name and version.
    
    Args:
      name: Template name
      version: Template version (latest if not specified)
    
    Returns:
      The template if found, None otherwise
    """
    return self.storage.get_template_by_name(name, version)
  
  def list_templates(
      self,
      tags: Optional[List[str]] = None,
      agent_type: Optional[str] = None,
      author: Optional[str] = None
  ) -> List[AgentTemplate]:
    """List templates with optional filtering.
    
    Args:
      tags: Filter by tags
      agent_type: Filter by agent type
      author: Filter by author
    
    Returns:
      List of matching templates
    """
    return self.storage.list_templates(
        tags=tags,
        agent_type=agent_type,
        author=author
    )
  
  def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
    """Update an existing template.
    
    Args:
      template_id: The template ID to update
      updates: Dictionary of fields to update
    
    Returns:
      True if update was successful, False otherwise
    """
    template = self.storage.get_template(template_id)
    if not template:
      return False
    
    # Create updated template
    updated_data = template.to_dict()
    updated_data.update(updates)
    updated_template = AgentTemplate.from_dict(updated_data)
    
    # Validate the updated template
    validation_result = self.validator.validate_template(updated_template)
    if not validation_result.is_valid:
      raise ValueError(f"Template validation failed: {validation_result.errors}")
    
    # Update storage
    success = self.storage.update_template(template_id, updated_template)
    
    if success:
      logger.info(f"Updated template {template_id}")
    else:
      logger.error(f"Failed to update template {template_id}")
    
    return success
  
  def delete_template(self, template_id: str) -> bool:
    """Delete a template.
    
    Args:
      template_id: The template ID to delete
    
    Returns:
      True if deletion was successful, False otherwise
    """
    template = self.storage.get_template(template_id)
    if not template:
      return False
    
    success = self.storage.delete_template(template_id)
    
    if success:
      logger.info(f"Deleted template {template.name}:{template.version}")
    else:
      logger.error(f"Failed to delete template {template_id}")
    
    return success
  
  def instantiate_agent(
      self,
      template_id: str,
      agent_name: Optional[str] = None,
      config_overrides: Optional[Dict[str, Any]] = None
  ) -> str:
    """Instantiate an agent from a template.
    
    Args:
      template_id: The template ID to instantiate from
      agent_name: Name for the agent instance
      config_overrides: Configuration overrides
    
    Returns:
      The agent ID of the created instance
    
    Raises:
      ValueError: If template not found or instantiation fails
    """
    template = self.storage.get_template(template_id)
    if not template:
      raise ValueError(f"Template {template_id} not found")
    
    # Update usage statistics
    template.update_usage()
    self.storage.update_template(template_id, template)
    
    # Get agent class
    agent_class = self._agent_type_map.get(template.agent_type)
    if not agent_class:
      raise ValueError(f"Unknown agent type: {template.agent_type}")
    
    # Prepare configuration
    config = template.agent_config.copy()
    if config_overrides:
      config.update(config_overrides)
    
    # Set agent name
    if agent_name:
      config['name'] = agent_name
    elif 'name' not in config:
      config['name'] = f"{template.name}_instance"
    
    try:
      # Instantiate the agent
      agent = agent_class(**config)
      
      # Generate agent ID (in a real implementation, this would be managed by the registry)
      agent_id = f"{template.name}_{template.id[:8]}_{id(agent)}"
      
      logger.info(f"Instantiated agent {agent_id} from template {template.name}")
      return agent_id
      
    except Exception as e:
      logger.error(f"Failed to instantiate agent from template {template_id}: {e}")
      raise ValueError(f"Agent instantiation failed: {e}")
  
  def create_template_from_agent(
      self,
      agent: BaseAgent,
      template_name: str,
      description: str = "",
      tags: Optional[List[str]] = None,
      author: Optional[str] = None
  ) -> str:
    """Create a template from an existing agent instance.
    
    Args:
      agent: The agent to create a template from
      template_name: Name for the template
      description: Description of the template
      tags: Tags for the template
      author: Author of the template
    
    Returns:
      The ID of the created template
    """
    # Extract agent configuration
    agent_config = {
        'name': agent.name,
        'description': agent.description,
    }
    
    # Add agent-specific configuration
    if isinstance(agent, LlmAgent):
      agent_config.update({
          'model': getattr(agent, 'model', None),
          'instruction': getattr(agent, 'instruction', None),
          'global_instruction': getattr(agent, 'global_instruction', None),
          'tools': [tool.__name__ if callable(tool) else str(tool) for tool in getattr(agent, 'tools', [])],
      })
    
    # Determine agent type
    agent_type = agent.__class__.__name__
    
    # Create template
    template = AgentTemplate(
        name=template_name,
        agent_type=agent_type,
        description=description,
        agent_config=agent_config,
        tags=tags or [],
        author=author
    )
    
    return self.create_template(template)
  
  def clone_template(
      self,
      template_id: str,
      new_name: str,
      modifications: Optional[Dict[str, Any]] = None
  ) -> str:
    """Clone an existing template with optional modifications.
    
    Args:
      template_id: The template ID to clone
      new_name: Name for the cloned template
      modifications: Optional modifications to apply
    
    Returns:
      The ID of the cloned template
    
    Raises:
      ValueError: If source template not found
    """
    source_template = self.storage.get_template(template_id)
    if not source_template:
      raise ValueError(f"Source template {template_id} not found")
    
    # Create cloned template data
    cloned_data = source_template.to_dict()
    cloned_data['name'] = new_name
    cloned_data['parent_template_id'] = template_id
    cloned_data['id'] = None  # Will be generated
    
    # Apply modifications
    if modifications:
      cloned_data.update(modifications)
    
    # Create new template
    cloned_template = AgentTemplate.from_dict(cloned_data)
    
    return self.create_template(cloned_template)
  
  def get_template_hierarchy(self, template_id: str) -> List[AgentTemplate]:
    """Get the inheritance hierarchy for a template.
    
    Args:
      template_id: The template ID
    
    Returns:
      List of templates in the inheritance chain (child to root)
    """
    hierarchy = []
    current_id = template_id
    
    while current_id:
      template = self.storage.get_template(current_id)
      if not template:
        break
      
      hierarchy.append(template)
      current_id = template.parent_template_id
    
    return hierarchy
  
  def search_templates(self, query: str) -> List[AgentTemplate]:
    """Search templates by name, description, or tags.
    
    Args:
      query: Search query string
    
    Returns:
      List of matching templates
    """
    return self.storage.search_templates(query)
  
  def get_template_statistics(self) -> Dict[str, Any]:
    """Get statistics about stored templates.
    
    Returns:
      Dictionary with template statistics
    """
    templates = self.storage.list_templates()
    
    stats = {
        'total_templates': len(templates),
        'by_type': {},
        'by_author': {},
        'total_usage': 0,
        'most_used': None,
        'recent_templates': []
    }
    
    for template in templates:
      # Count by type
      agent_type = template.agent_type
      stats['by_type'][agent_type] = stats['by_type'].get(agent_type, 0) + 1
      
      # Count by author
      author = template.author or 'unknown'
      stats['by_author'][author] = stats['by_author'].get(author, 0) + 1
      
      # Track usage
      stats['total_usage'] += template.usage_count
      
      # Find most used
      if not stats['most_used'] or template.usage_count > stats['most_used']['usage_count']:
        stats['most_used'] = {
            'name': template.name,
            'usage_count': template.usage_count
        }
    
    # Get recent templates (last 5)
    recent = sorted(templates, key=lambda t: t.created_at, reverse=True)[:5]
    stats['recent_templates'] = [
        {'name': t.name, 'created_at': t.created_at.isoformat()}
        for t in recent
    ]
    
    return stats

