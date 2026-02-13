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

"""Agent template model for storing and managing agent configurations."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class AgentTemplate(BaseModel):
  """Template for creating and managing agent configurations.
  
  This model stores all the necessary information to instantiate an agent,
  including its configuration, dependencies, and metadata.
  """
  
  # Core identification
  id: str = Field(default_factory=lambda: str(uuid.uuid4()))
  """Unique identifier for the template."""
  
  name: str
  """Human-readable name for the template."""
  
  version: str = "1.0.0"
  """Semantic version of the template."""
  
  description: str = ""
  """Description of what this agent template does."""
  
  # Agent configuration
  agent_type: str
  """Type of agent (LlmAgent, SequentialAgent, ParallelAgent, LoopAgent)."""
  
  agent_config: Dict[str, Any]
  """Complete agent configuration dictionary."""
  
  # Dependencies and requirements
  required_tools: List[str] = Field(default_factory=list)
  """List of required tool names/types."""
  
  required_environments: List[str] = Field(default_factory=list)
  """List of supported environments (local, wsl2, ssh)."""
  
  dependencies: List[str] = Field(default_factory=list)
  """List of other template IDs this template depends on."""
  
  # Metadata
  tags: List[str] = Field(default_factory=list)
  """Tags for categorizing and searching templates."""
  
  author: Optional[str] = None
  """Author of the template."""
  
  created_at: datetime = Field(default_factory=datetime.utcnow)
  """When the template was created."""
  
  updated_at: datetime = Field(default_factory=datetime.utcnow)
  """When the template was last updated."""
  
  # Template inheritance
  parent_template_id: Optional[str] = None
  """ID of parent template if this inherits from another."""
  
  # Validation and constraints
  min_adk_version: Optional[str] = None
  """Minimum ADK version required."""
  
  max_adk_version: Optional[str] = None
  """Maximum ADK version supported."""
  
  # Usage statistics
  usage_count: int = 0
  """Number of times this template has been instantiated."""
  
  last_used: Optional[datetime] = None
  """When this template was last used."""
  
  def update_usage(self) -> None:
    """Update usage statistics."""
    self.usage_count += 1
    self.last_used = datetime.utcnow()
    self.updated_at = datetime.utcnow()
  
  def is_compatible_with_environment(self, environment: str) -> bool:
    """Check if template is compatible with given environment."""
    if not self.required_environments:
      return True  # No specific requirements
    return environment in self.required_environments
  
  def get_full_name(self) -> str:
    """Get full template name including version."""
    return f"{self.name}:{self.version}"
  
  def to_dict(self) -> Dict[str, Any]:
    """Convert template to dictionary for serialization."""
    return self.model_dump()
  
  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> AgentTemplate:
    """Create template from dictionary."""
    return cls.model_validate(data)

