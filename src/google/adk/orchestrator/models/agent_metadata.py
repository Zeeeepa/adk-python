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

"""Agent metadata model for runtime agent information."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class AgentStatus(str, Enum):
  """Status of an agent instance."""
  CREATED = "created"
  INITIALIZING = "initializing"
  READY = "ready"
  RUNNING = "running"
  PAUSED = "paused"
  COMPLETED = "completed"
  FAILED = "failed"
  TERMINATED = "terminated"


class AgentCapability(BaseModel):
  """Represents a capability of an agent."""
  name: str
  description: str
  parameters: Dict[str, Any] = Field(default_factory=dict)
  required: bool = False


class AgentMetadata(BaseModel):
  """Runtime metadata for agent instances.
  
  This model tracks the runtime state and capabilities of agent instances
  created from templates.
  """
  
  # Core identification
  agent_id: str
  """Unique identifier for the agent instance."""
  
  template_id: str
  """ID of the template this agent was created from."""
  
  name: str
  """Runtime name of the agent instance."""
  
  # Status and lifecycle
  status: AgentStatus = AgentStatus.CREATED
  """Current status of the agent."""
  
  created_at: datetime = Field(default_factory=datetime.utcnow)
  """When the agent instance was created."""
  
  started_at: Optional[datetime] = None
  """When the agent started running."""
  
  completed_at: Optional[datetime] = None
  """When the agent completed execution."""
  
  # Runtime configuration
  environment: str
  """Environment the agent is running in (local, wsl2, ssh)."""
  
  execution_context: Dict[str, Any] = Field(default_factory=dict)
  """Runtime execution context and state."""
  
  # Capabilities and tools
  capabilities: List[AgentCapability] = Field(default_factory=list)
  """List of agent capabilities."""
  
  active_tools: List[str] = Field(default_factory=list)
  """List of currently active tool names."""
  
  # Relationships
  parent_agent_id: Optional[str] = None
  """ID of parent agent if this is a sub-agent."""
  
  sub_agent_ids: List[str] = Field(default_factory=list)
  """IDs of sub-agents managed by this agent."""
  
  # Performance metrics
  execution_time: Optional[float] = None
  """Total execution time in seconds."""
  
  memory_usage: Optional[int] = None
  """Peak memory usage in bytes."""
  
  api_calls_made: int = 0
  """Number of API calls made by this agent."""
  
  # Error handling
  last_error: Optional[str] = None
  """Last error message if any."""
  
  error_count: int = 0
  """Total number of errors encountered."""
  
  # Tracing and debugging
  trace_id: Optional[str] = None
  """Trace ID for distributed tracing."""
  
  debug_enabled: bool = False
  """Whether debug mode is enabled."""
  
  log_level: str = "INFO"
  """Current log level."""
  
  def update_status(self, status: AgentStatus, error: Optional[str] = None) -> None:
    """Update agent status and related timestamps."""
    self.status = status
    
    if status == AgentStatus.RUNNING and not self.started_at:
      self.started_at = datetime.utcnow()
    elif status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TERMINATED]:
      self.completed_at = datetime.utcnow()
      if self.started_at:
        self.execution_time = (self.completed_at - self.started_at).total_seconds()
    
    if error:
      self.last_error = error
      self.error_count += 1
  
  def add_capability(self, name: str, description: str, 
                    parameters: Optional[Dict[str, Any]] = None,
                    required: bool = False) -> None:
    """Add a capability to the agent."""
    capability = AgentCapability(
        name=name,
        description=description,
        parameters=parameters or {},
        required=required
    )
    self.capabilities.append(capability)
  
  def has_capability(self, name: str) -> bool:
    """Check if agent has a specific capability."""
    return any(cap.name == name for cap in self.capabilities)
  
  def get_capability(self, name: str) -> Optional[AgentCapability]:
    """Get a specific capability by name."""
    for cap in self.capabilities:
      if cap.name == name:
        return cap
    return None
  
  def is_running(self) -> bool:
    """Check if agent is currently running."""
    return self.status == AgentStatus.RUNNING
  
  def is_completed(self) -> bool:
    """Check if agent has completed execution."""
    return self.status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TERMINATED]
  
  def get_runtime_summary(self) -> Dict[str, Any]:
    """Get a summary of runtime information."""
    return {
        "agent_id": self.agent_id,
        "name": self.name,
        "status": self.status.value,
        "environment": self.environment,
        "execution_time": self.execution_time,
        "api_calls_made": self.api_calls_made,
        "error_count": self.error_count,
        "capabilities_count": len(self.capabilities),
        "sub_agents_count": len(self.sub_agent_ids),
    }

