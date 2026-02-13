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

"""Template storage implementation for agent templates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..models.agent_template import AgentTemplate

logger = logging.getLogger('google_adk.orchestrator.storage.template_storage')


class TemplateStorage:
  """Storage backend for agent templates.
  
  This class provides persistent storage for agent templates with support
  for both in-memory and file-based storage backends.
  """
  
  def __init__(self, storage_path: Optional[str] = None):
    """Initialize template storage.
    
    Args:
      storage_path: Path for file-based storage (None for in-memory)
    """
    self.storage_path = Path(storage_path) if storage_path else None
    self._templates: Dict[str, AgentTemplate] = {}
    self._name_index: Dict[str, Dict[str, str]] = {}  # name -> version -> id
    
    if self.storage_path:
      self.storage_path.mkdir(parents=True, exist_ok=True)
      self._load_templates_from_disk()
    
    logger.info(f"Template storage initialized ({'file-based' if storage_path else 'in-memory'})")
  
  def store_template(self, template: AgentTemplate) -> str:
    """Store a template.
    
    Args:
      template: Template to store
    
    Returns:
      Template ID
    """
    template_id = template.id
    
    # Store in memory
    self._templates[template_id] = template
    
    # Update name index
    if template.name not in self._name_index:
      self._name_index[template.name] = {}
    self._name_index[template.name][template.version] = template_id
    
    # Store to disk if file-based storage
    if self.storage_path:
      self._save_template_to_disk(template)
    
    logger.debug(f"Stored template {template.name}:{template.version} with ID {template_id}")
    return template_id
  
  def get_template(self, template_id: str) -> Optional[AgentTemplate]:
    """Get a template by ID.
    
    Args:
      template_id: Template ID
    
    Returns:
      Template if found, None otherwise
    """
    return self._templates.get(template_id)
  
  def get_template_by_name(
      self,
      name: str,
      version: Optional[str] = None
  ) -> Optional[AgentTemplate]:
    """Get a template by name and version.
    
    Args:
      name: Template name
      version: Template version (latest if not specified)
    
    Returns:
      Template if found, None otherwise
    """
    if name not in self._name_index:
      return None
    
    versions = self._name_index[name]
    
    if version:
      # Get specific version
      template_id = versions.get(version)
      if template_id:
        return self._templates.get(template_id)
    else:
      # Get latest version (highest semantic version)
      if not versions:
        return None
      
      # Simple version sorting (for production, use proper semantic versioning)
      latest_version = max(versions.keys())
      template_id = versions[latest_version]
      return self._templates.get(template_id)
    
    return None
  
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
    templates = list(self._templates.values())
    
    # Apply filters
    if tags:
      templates = [t for t in templates if any(tag in t.tags for tag in tags)]
    
    if agent_type:
      templates = [t for t in templates if t.agent_type == agent_type]
    
    if author:
      templates = [t for t in templates if t.author == author]
    
    return templates
  
  def update_template(self, template_id: str, template: AgentTemplate) -> bool:
    """Update an existing template.
    
    Args:
      template_id: Template ID to update
      template: Updated template
    
    Returns:
      True if update was successful, False otherwise
    """
    if template_id not in self._templates:
      return False
    
    old_template = self._templates[template_id]
    
    # Update in memory
    self._templates[template_id] = template
    
    # Update name index if name or version changed
    if old_template.name != template.name or old_template.version != template.version:
      # Remove old index entry
      if old_template.name in self._name_index:
        self._name_index[old_template.name].pop(old_template.version, None)
        if not self._name_index[old_template.name]:
          del self._name_index[old_template.name]
      
      # Add new index entry
      if template.name not in self._name_index:
        self._name_index[template.name] = {}
      self._name_index[template.name][template.version] = template_id
    
    # Update on disk if file-based storage
    if self.storage_path:
      self._save_template_to_disk(template)
      # Remove old file if name changed
      if old_template.name != template.name:
        old_file = self.storage_path / f"{old_template.name}_{old_template.version}.json"
        if old_file.exists():
          old_file.unlink()
    
    logger.debug(f"Updated template {template_id}")
    return True
  
  def delete_template(self, template_id: str) -> bool:
    """Delete a template.
    
    Args:
      template_id: Template ID to delete
    
    Returns:
      True if deletion was successful, False otherwise
    """
    if template_id not in self._templates:
      return False
    
    template = self._templates[template_id]
    
    # Remove from memory
    del self._templates[template_id]
    
    # Remove from name index
    if template.name in self._name_index:
      self._name_index[template.name].pop(template.version, None)
      if not self._name_index[template.name]:
        del self._name_index[template.name]
    
    # Remove from disk if file-based storage
    if self.storage_path:
      template_file = self.storage_path / f"{template.name}_{template.version}.json"
      if template_file.exists():
        template_file.unlink()
    
    logger.debug(f"Deleted template {template.name}:{template.version}")
    return True
  
  def search_templates(self, query: str) -> List[AgentTemplate]:
    """Search templates by name, description, or tags.
    
    Args:
      query: Search query string
    
    Returns:
      List of matching templates
    """
    query_lower = query.lower()
    matching_templates = []
    
    for template in self._templates.values():
      # Search in name
      if query_lower in template.name.lower():
        matching_templates.append(template)
        continue
      
      # Search in description
      if query_lower in template.description.lower():
        matching_templates.append(template)
        continue
      
      # Search in tags
      if any(query_lower in tag.lower() for tag in template.tags):
        matching_templates.append(template)
        continue
    
    return matching_templates
  
  def _save_template_to_disk(self, template: AgentTemplate) -> None:
    """Save a template to disk.
    
    Args:
      template: Template to save
    """
    if not self.storage_path:
      return
    
    try:
      template_file = self.storage_path / f"{template.name}_{template.version}.json"
      template_data = template.to_dict()
      
      with open(template_file, 'w', encoding='utf-8') as f:
        json.dump(template_data, f, indent=2, default=str)
      
      logger.debug(f"Saved template to disk: {template_file}")
      
    except Exception as e:
      logger.error(f"Failed to save template to disk: {e}")
  
  def _load_templates_from_disk(self) -> None:
    """Load templates from disk.
    """
    if not self.storage_path or not self.storage_path.exists():
      return
    
    try:
      template_files = list(self.storage_path.glob("*.json"))
      loaded_count = 0
      
      for template_file in template_files:
        try:
          with open(template_file, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
          
          template = AgentTemplate.from_dict(template_data)
          
          # Store in memory
          self._templates[template.id] = template
          
          # Update name index
          if template.name not in self._name_index:
            self._name_index[template.name] = {}
          self._name_index[template.name][template.version] = template.id
          
          loaded_count += 1
          
        except Exception as e:
          logger.error(f"Failed to load template from {template_file}: {e}")
      
      logger.info(f"Loaded {loaded_count} templates from disk")
      
    except Exception as e:
      logger.error(f"Failed to load templates from disk: {e}")
  
  def get_storage_info(self) -> Dict[str, Any]:
    """Get information about the storage backend.
    
    Returns:
      Dictionary with storage information
    """
    return {
        'type': 'file-based' if self.storage_path else 'in-memory',
        'path': str(self.storage_path) if self.storage_path else None,
        'template_count': len(self._templates),
        'unique_names': len(self._name_index)
    }

