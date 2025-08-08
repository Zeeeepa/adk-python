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

"""Execution context model for tracking agent execution state."""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from pydantic import BaseModel
from pydantic import Field


class ExecutionMode(str, Enum):
  """Mode of execution for agents."""
  SYNC = "sync"
  ASYNC = "async"
  LIVE = "live"
  BATCH = "batch"


class ExecutionPriority(str, Enum):
  """Priority levels for execution."""
  LOW = "low"
  NORMAL = "normal"
  HIGH = "high"
  CRITICAL = "critical"


class ExecutionContext(BaseModel):
  """Context for tracking and managing agent execution.
  
  This model maintains the execution state, resources, and coordination
  information for agent instances during their lifecycle.
  """
  
  # Core execution info
  execution_id: str
  """Unique identifier for this execution context."""
  
  agent_id: str
  """ID of the agent being executed."""
  
  session_id: Optional[str] = None
  """Session ID if part of a larger session."""
  
  # Execution configuration
  mode: ExecutionMode = ExecutionMode.ASYNC
  """Execution mode (sync, async, live, batch)."""
  
  priority: ExecutionPriority = ExecutionPriority.NORMAL
  """Execution priority level."""
  
  timeout: Optional[float] = None
  """Execution timeout in seconds."""
  
  max_retries: int = 3
  """Maximum number of retry attempts."""
  
  # State management
  state: Dict[str, Any] = Field(default_factory=dict)
  """Execution state dictionary."""
  
  shared_state: Dict[str, Any] = Field(default_factory=dict)
  """State shared across multiple agents."""
  
  artifacts: Dict[str, Any] = Field(default_factory=dict)
  """Execution artifacts and outputs."""
  
  # Resource management
  allocated_resources: Dict[str, Any] = Field(default_factory=dict)
  """Resources allocated to this execution."""
  
  resource_limits: Dict[str, Any] = Field(default_factory=dict)
  """Resource limits for this execution."""
  
  # Coordination and dependencies
  dependencies: Set[str] = Field(default_factory=set)
  """Set of execution IDs this execution depends on."""
  
  dependents: Set[str] = Field(default_factory=set)
  """Set of execution IDs that depend on this execution."""
  
  blocking_on: Set[str] = Field(default_factory=set)
  """Set of execution IDs this execution is currently blocked on."""
  
  # Execution tracking
  started_at: Optional[datetime] = None
  """When execution started."""
  
  completed_at: Optional[datetime] = None
  """When execution completed."""
  
  last_heartbeat: Optional[datetime] = None
  """Last heartbeat timestamp."""
  
  # Progress tracking
  progress_percentage: float = 0.0
  """Execution progress as percentage (0-100)."""
  
  current_step: Optional[str] = None
  """Current execution step description."""
  
  total_steps: Optional[int] = None
  """Total number of steps if known."""
  
  completed_steps: int = 0
  """Number of completed steps."""
  
  # Error handling and retries
  retry_count: int = 0
  """Current retry attempt count."""
  
  last_error: Optional[str] = None
  """Last error message."""
  
  error_history: List[str] = Field(default_factory=list)
  """History of error messages."""
  
  # Cancellation and cleanup
  cancellation_requested: bool = False
  """Whether cancellation has been requested."""
  
  cancellation_reason: Optional[str] = None
  """Reason for cancellation if any."""
  
  cleanup_required: bool = False
  """Whether cleanup is required after execution."""
  
  # Async execution support
  _task: Optional[asyncio.Task] = Field(default=None, exclude=True)
  """Async task reference (not serialized)."""
  
  _future: Optional[asyncio.Future] = Field(default=None, exclude=True)
  """Future reference for result (not serialized)."""
  
  def start_execution(self) -> None:
    """Mark execution as started."""
    self.started_at = datetime.utcnow()
    self.last_heartbeat = self.started_at
  
  def complete_execution(self, success: bool = True, error: Optional[str] = None) -> None:
    """Mark execution as completed."""
    self.completed_at = datetime.utcnow()
    self.progress_percentage = 100.0
    
    if not success and error:
      self.last_error = error
      self.error_history.append(error)
  
  def update_progress(self, percentage: float, step: Optional[str] = None) -> None:
    """Update execution progress."""
    self.progress_percentage = max(0.0, min(100.0, percentage))
    self.last_heartbeat = datetime.utcnow()
    
    if step:
      self.current_step = step
      if self.current_step != getattr(self, '_last_step', None):
        self.completed_steps += 1
        self._last_step = self.current_step
  
  def heartbeat(self) -> None:
    """Update heartbeat timestamp."""
    self.last_heartbeat = datetime.utcnow()
  
  def request_cancellation(self, reason: Optional[str] = None) -> None:
    """Request cancellation of execution."""
    self.cancellation_requested = True
    self.cancellation_reason = reason
  
  def add_dependency(self, execution_id: str) -> None:
    """Add a dependency on another execution."""
    self.dependencies.add(execution_id)
  
  def remove_dependency(self, execution_id: str) -> None:
    """Remove a dependency."""
    self.dependencies.discard(execution_id)
    self.blocking_on.discard(execution_id)
  
  def add_dependent(self, execution_id: str) -> None:
    """Add an execution that depends on this one."""
    self.dependents.add(execution_id)
  
  def is_blocked(self) -> bool:
    """Check if execution is blocked by dependencies."""
    return len(self.blocking_on) > 0
  
  def can_start(self) -> bool:
    """Check if execution can start (no blocking dependencies)."""
    return not self.is_blocked() and not self.cancellation_requested
  
  def is_running(self) -> bool:
    """Check if execution is currently running."""
    return self.started_at is not None and self.completed_at is None
  
  def is_completed(self) -> bool:
    """Check if execution is completed."""
    return self.completed_at is not None
  
  def get_execution_time(self) -> Optional[float]:
    """Get execution time in seconds."""
    if not self.started_at:
      return None
    
    end_time = self.completed_at or datetime.utcnow()
    return (end_time - self.started_at).total_seconds()
  
  def get_status_summary(self) -> Dict[str, Any]:
    """Get a summary of execution status."""
    return {
        "execution_id": self.execution_id,
        "agent_id": self.agent_id,
        "mode": self.mode.value,
        "priority": self.priority.value,
        "progress": self.progress_percentage,
        "current_step": self.current_step,
        "is_running": self.is_running(),
        "is_completed": self.is_completed(),
        "is_blocked": self.is_blocked(),
        "execution_time": self.get_execution_time(),
        "retry_count": self.retry_count,
        "error_count": len(self.error_history),
    }

