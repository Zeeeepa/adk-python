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

"""Execution tracker for monitoring and tracing agent activities."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .models.execution_context import ExecutionContext
from .models.execution_context import ExecutionMode
from .models.execution_context import ExecutionPriority

logger = logging.getLogger('google_adk.orchestrator.execution_tracker')


class ExecutionTracker:
  """Tracker for monitoring and managing agent execution.
  
  The ExecutionTracker provides comprehensive monitoring of agent executions
  including status tracking, performance metrics, error handling, and
  distributed tracing capabilities.
  """
  
  def __init__(self, max_concurrent_executions: int = 50):
    """Initialize the execution tracker.
    
    Args:
      max_concurrent_executions: Maximum number of concurrent executions
    """
    self.max_concurrent_executions = max_concurrent_executions
    self._executions: Dict[str, ExecutionContext] = {}
    self._execution_history: List[ExecutionContext] = []
    self._active_tasks: Dict[str, asyncio.Task] = {}
    self._lock = asyncio.Lock()
    
    logger.info(f"Execution tracker initialized (max concurrent: {max_concurrent_executions})")
  
  async def create_execution_context(
      self,
      agent_id: str,
      mode: ExecutionMode = ExecutionMode.ASYNC,
      priority: ExecutionPriority = ExecutionPriority.NORMAL,
      timeout: Optional[float] = None,
      session_id: Optional[str] = None
  ) -> str:
    """Create a new execution context.
    
    Args:
      agent_id: ID of the agent to execute
      mode: Execution mode
      priority: Execution priority
      timeout: Execution timeout in seconds
      session_id: Optional session ID
    
    Returns:
      Execution ID for the created context
    """
    execution_id = str(uuid.uuid4())
    
    context = ExecutionContext(
        execution_id=execution_id,
        agent_id=agent_id,
        session_id=session_id,
        mode=mode,
        priority=priority,
        timeout=timeout
    )
    
    async with self._lock:
      self._executions[execution_id] = context
    
    logger.info(f"Created execution context {execution_id} for agent {agent_id}")
    return execution_id
  
  async def start_execution(self, execution_id: str) -> bool:
    """Start an execution.
    
    Args:
      execution_id: ID of the execution to start
    
    Returns:
      True if execution was started, False otherwise
    """
    async with self._lock:
      context = self._executions.get(execution_id)
      if not context:
        logger.error(f"Execution context {execution_id} not found")
        return False
      
      if not context.can_start():
        logger.warning(f"Execution {execution_id} cannot start (blocked or cancelled)")
        return False
      
      # Check concurrent execution limit
      active_count = sum(1 for ctx in self._executions.values() if ctx.is_running())
      if active_count >= self.max_concurrent_executions:
        logger.warning(f"Maximum concurrent executions reached ({self.max_concurrent_executions})")
        return False
      
      context.start_execution()
      logger.info(f"Started execution {execution_id}")
      return True
  
  async def complete_execution(
      self,
      execution_id: str,
      success: bool = True,
      error: Optional[str] = None,
      artifacts: Optional[Dict[str, Any]] = None
  ) -> bool:
    """Complete an execution.
    
    Args:
      execution_id: ID of the execution to complete
      success: Whether execution was successful
      error: Error message if execution failed
      artifacts: Execution artifacts
    
    Returns:
      True if execution was completed, False otherwise
    """
    async with self._lock:
      context = self._executions.get(execution_id)
      if not context:
        logger.error(f"Execution context {execution_id} not found")
        return False
      
      context.complete_execution(success=success, error=error)
      
      if artifacts:
        context.artifacts.update(artifacts)
      
      # Move to history
      self._execution_history.append(context)
      
      # Clean up active task
      if execution_id in self._active_tasks:
        task = self._active_tasks.pop(execution_id)
        if not task.done():
          task.cancel()
      
      logger.info(f"Completed execution {execution_id} (success: {success})")
      return True
  
  async def cancel_execution(self, execution_id: str, reason: Optional[str] = None) -> bool:
    """Cancel an execution.
    
    Args:
      execution_id: ID of the execution to cancel
      reason: Reason for cancellation
    
    Returns:
      True if execution was cancelled, False otherwise
    """
    async with self._lock:
      context = self._executions.get(execution_id)
      if not context:
        logger.error(f"Execution context {execution_id} not found")
        return False
      
      context.request_cancellation(reason)
      
      # Cancel active task
      if execution_id in self._active_tasks:
        task = self._active_tasks[execution_id]
        task.cancel()
        logger.info(f"Cancelled execution {execution_id}: {reason or 'No reason provided'}")
      
      return True
  
  async def update_execution_progress(
      self,
      execution_id: str,
      percentage: float,
      step: Optional[str] = None
  ) -> bool:
    """Update execution progress.
    
    Args:
      execution_id: ID of the execution
      percentage: Progress percentage (0-100)
      step: Current step description
    
    Returns:
      True if progress was updated, False otherwise
    """
    async with self._lock:
      context = self._executions.get(execution_id)
      if not context:
        return False
      
      context.update_progress(percentage, step)
      return True
  
  async def add_execution_dependency(
      self,
      execution_id: str,
      dependency_id: str
  ) -> bool:
    """Add a dependency between executions.
    
    Args:
      execution_id: ID of the dependent execution
      dependency_id: ID of the execution to depend on
    
    Returns:
      True if dependency was added, False otherwise
    """
    async with self._lock:
      context = self._executions.get(execution_id)
      dependency_context = self._executions.get(dependency_id)
      
      if not context or not dependency_context:
        return False
      
      context.add_dependency(dependency_id)
      dependency_context.add_dependent(execution_id)
      
      # If dependency is not completed, add to blocking set
      if not dependency_context.is_completed():
        context.blocking_on.add(dependency_id)
      
      logger.info(f"Added dependency: {execution_id} depends on {dependency_id}")
      return True
  
  async def resolve_execution_dependency(self, dependency_id: str) -> None:
    """Resolve a dependency when an execution completes.
    
    Args:
      dependency_id: ID of the completed execution
    """
    async with self._lock:
      dependency_context = self._executions.get(dependency_id)
      if not dependency_context:
        return
      
      # Update all dependent executions
      for dependent_id in dependency_context.dependents:
        dependent_context = self._executions.get(dependent_id)
        if dependent_context:
          dependent_context.remove_dependency(dependency_id)
          logger.debug(f"Resolved dependency for {dependent_id}")
  
  def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
    """Get the status of an execution.
    
    Args:
      execution_id: ID of the execution
    
    Returns:
      Execution status information or None if not found
    """
    context = self._executions.get(execution_id)
    if context:
      return context.get_status_summary()
    return None
  
  def get_active_executions(self) -> List[Dict[str, Any]]:
    """Get all active executions.
    
    Returns:
      List of active execution summaries
    """
    return [
        ctx.get_status_summary()
        for ctx in self._executions.values()
        if ctx.is_running()
    ]
  
  def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
    """Get execution history.
    
    Args:
      limit: Maximum number of historical executions to return
    
    Returns:
      List of historical execution summaries
    """
    # Return most recent executions first
    recent_history = self._execution_history[-limit:] if limit > 0 else self._execution_history
    return [ctx.get_status_summary() for ctx in reversed(recent_history)]
  
  def get_execution_statistics(self) -> Dict[str, Any]:
    """Get execution statistics.
    
    Returns:
      Dictionary with execution statistics
    """
    active_executions = [ctx for ctx in self._executions.values() if ctx.is_running()]
    completed_executions = self._execution_history
    
    # Calculate success rate
    total_completed = len(completed_executions)
    successful_completed = sum(1 for ctx in completed_executions if ctx.last_error is None)
    success_rate = (successful_completed / total_completed * 100) if total_completed > 0 else 0
    
    # Calculate average execution time
    execution_times = [ctx.get_execution_time() for ctx in completed_executions if ctx.get_execution_time()]
    avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
    
    return {
        'active_executions': len(active_executions),
        'total_completed': total_completed,
        'success_rate': round(success_rate, 2),
        'average_execution_time': round(avg_execution_time, 2),
        'max_concurrent_limit': self.max_concurrent_executions,
        'current_utilization': round(len(active_executions) / self.max_concurrent_executions * 100, 2)
    }
  
  async def execute_agent_async(
      self,
      agent_id: str,
      input_data: Dict[str, Any],
      timeout: Optional[float] = None,
      environment: str = "local"
  ) -> str:
    """Execute an agent asynchronously with full tracking.
    
    Args:
      agent_id: ID of the agent to execute
      input_data: Input data for the agent
      timeout: Execution timeout in seconds
      environment: Execution environment
    
    Returns:
      Execution ID for tracking
    """
    # Create execution context
    execution_id = await self.create_execution_context(
        agent_id=agent_id,
        mode=ExecutionMode.ASYNC,
        timeout=timeout
    )
    
    # Create and start async task
    task = asyncio.create_task(
        self._execute_agent_task(execution_id, agent_id, input_data, environment)
    )
    
    async with self._lock:
      self._active_tasks[execution_id] = task
    
    # Start execution tracking
    await self.start_execution(execution_id)
    
    return execution_id
  
  async def _execute_agent_task(
      self,
      execution_id: str,
      agent_id: str,
      input_data: Dict[str, Any],
      environment: str
  ) -> None:
    """Internal task for executing an agent.
    
    Args:
      execution_id: Execution ID
      agent_id: Agent ID
      input_data: Input data
      environment: Environment name
    """
    try:
      # This is a placeholder for actual agent execution
      # In a real implementation, this would:
      # 1. Get the agent instance from the registry
      # 2. Set up the execution environment
      # 3. Execute the agent with proper monitoring
      # 4. Collect results and artifacts
      
      logger.info(f"Executing agent {agent_id} in environment {environment}")
      
      # Simulate execution with progress updates
      for i in range(5):
        await asyncio.sleep(1)  # Simulate work
        await self.update_execution_progress(
            execution_id,
            (i + 1) * 20,
            f"Step {i + 1} of 5"
        )
      
      # Complete execution
      await self.complete_execution(
          execution_id,
          success=True,
          artifacts={"result": "Agent execution completed successfully"}
      )
      
    except asyncio.CancelledError:
      await self.complete_execution(
          execution_id,
          success=False,
          error="Execution was cancelled"
      )
      raise
    except Exception as e:
      logger.error(f"Agent execution failed: {e}")
      await self.complete_execution(
          execution_id,
          success=False,
          error=str(e)
      )
  
  async def cleanup_completed_executions(self, max_history: int = 1000) -> int:
    """Clean up old completed executions.
    
    Args:
      max_history: Maximum number of historical executions to keep
    
    Returns:
      Number of executions cleaned up
    """
    if len(self._execution_history) <= max_history:
      return 0
    
    # Keep only the most recent executions
    excess_count = len(self._execution_history) - max_history
    self._execution_history = self._execution_history[excess_count:]
    
    logger.info(f"Cleaned up {excess_count} old execution records")
    return excess_count

