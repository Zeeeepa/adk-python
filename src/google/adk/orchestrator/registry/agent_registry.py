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

"""Agent registry for managing agent instances and metadata."""

from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..agents.base_agent import BaseAgent
from ..models.agent_metadata import AgentMetadata
from ..models.agent_metadata import AgentStatus

logger = logging.getLogger('google_adk.orchestrator.registry.agent_registry')


class AgentRegistry:
  """Registry for managing agent instances and their metadata.
  
  The AgentRegistry provides centralized management of agent instances
  including registration, discovery, lifecycle management, and metadata
  tracking.
  """
  
  def __init__(self):
    """Initialize the agent registry."""
    self._agents: Dict[str, BaseAgent] = {}
    self._metadata: Dict[str, AgentMetadata] = {}
    self._capability_index: Dict[str, List[str]] = {}  # capability -> agent_ids
    
    logger.info("Agent registry initialized")
  
  def register_agent(
      self,
      agent: BaseAgent,
      agent_id: Optional[str] = None,
      template_id: Optional[str] = None,
      environment: str = "local"
  ) -> str:
    """Register an agent instance.
    
    Args:
      agent: Agent instance to register
      agent_id: Optional agent ID (generated if not provided)
      template_id: ID of template used to create the agent
      environment: Environment the agent will run in
    
    Returns:
      Agent ID of the registered agent
    """
    if not agent_id:
      agent_id = f"{agent.name}_{id(agent)}"
    
    # Store agent instance
    self._agents[agent_id] = agent
    
    # Create metadata
    metadata = AgentMetadata(
        agent_id=agent_id,
        template_id=template_id or "unknown",
        name=agent.name,
        environment=environment,
        status=AgentStatus.CREATED
    )
    
    # Extract capabilities from agent
    self._extract_agent_capabilities(agent, metadata)
    
    self._metadata[agent_id] = metadata
    
    # Update capability index
    self._update_capability_index(agent_id, metadata)
    
    logger.info(f"Registered agent {agent.name} with ID {agent_id}")
    return agent_id
  
  def unregister_agent(self, agent_id: str) -> bool:
    """Unregister an agent instance.
    
    Args:
      agent_id: ID of the agent to unregister
    
    Returns:
      True if agent was unregistered, False if not found
    """
    if agent_id not in self._agents:
      return False
    
    # Remove from capability index
    metadata = self._metadata.get(agent_id)
    if metadata:
      for capability in metadata.capabilities:
        if capability.name in self._capability_index:
          self._capability_index[capability.name] = [
              aid for aid in self._capability_index[capability.name]
              if aid != agent_id
          ]
          if not self._capability_index[capability.name]:
            del self._capability_index[capability.name]
    
    # Remove agent and metadata
    del self._agents[agent_id]
    if agent_id in self._metadata:
      del self._metadata[agent_id]
    
    logger.info(f"Unregistered agent {agent_id}")
    return True
  
  def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
    """Get an agent instance by ID.
    
    Args:
      agent_id: Agent ID
    
    Returns:
      Agent instance if found, None otherwise
    """
    return self._agents.get(agent_id)
  
  def get_agent_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
    """Get agent metadata by ID.
    
    Args:
      agent_id: Agent ID
    
    Returns:
      Agent metadata if found, None otherwise
    """
    return self._metadata.get(agent_id)
  
  def list_agents(
      self,
      status: Optional[AgentStatus] = None,
      environment: Optional[str] = None,
      template_id: Optional[str] = None
  ) -> List[str]:
    """List agent IDs with optional filtering.
    
    Args:
      status: Filter by agent status
      environment: Filter by environment
      template_id: Filter by template ID
    
    Returns:
      List of matching agent IDs
    """
    agent_ids = []
    
    for agent_id, metadata in self._metadata.items():
      # Apply filters
      if status and metadata.status != status:
        continue
      
      if environment and metadata.environment != environment:
        continue
      
      if template_id and metadata.template_id != template_id:
        continue
      
      agent_ids.append(agent_id)
    
    return agent_ids
  
  def find_agents_by_capability(self, capability: str) -> List[str]:
    """Find agents that have a specific capability.
    
    Args:
      capability: Capability name to search for
    
    Returns:
      List of agent IDs that have the capability
    """
    return self._capability_index.get(capability, [])
  
  def find_agents_by_name(self, name: str) -> List[str]:
    """Find agents by name.
    
    Args:
      name: Agent name to search for
    
    Returns:
      List of agent IDs with matching names
    """
    return [
        agent_id for agent_id, metadata in self._metadata.items()
        if metadata.name == name
    ]
  
  def update_agent_status(
      self,
      agent_id: str,
      status: AgentStatus,
      error: Optional[str] = None
  ) -> bool:
    """Update agent status.
    
    Args:
      agent_id: Agent ID
      status: New status
      error: Error message if status indicates failure
    
    Returns:
      True if status was updated, False if agent not found
    """
    metadata = self._metadata.get(agent_id)
    if not metadata:
      return False
    
    metadata.update_status(status, error)
    logger.debug(f"Updated agent {agent_id} status to {status.value}")
    return True
  
  def add_agent_capability(
      self,
      agent_id: str,
      capability_name: str,
      description: str,
      parameters: Optional[Dict[str, Any]] = None,
      required: bool = False
  ) -> bool:
    """Add a capability to an agent.
    
    Args:
      agent_id: Agent ID
      capability_name: Name of the capability
      description: Description of the capability
      parameters: Capability parameters
      required: Whether the capability is required
    
    Returns:
      True if capability was added, False if agent not found
    """
    metadata = self._metadata.get(agent_id)
    if not metadata:
      return False
    
    metadata.add_capability(capability_name, description, parameters, required)
    
    # Update capability index
    if capability_name not in self._capability_index:
      self._capability_index[capability_name] = []
    if agent_id not in self._capability_index[capability_name]:
      self._capability_index[capability_name].append(agent_id)
    
    logger.debug(f"Added capability {capability_name} to agent {agent_id}")
    return True
  
  def get_registry_statistics(self) -> Dict[str, Any]:
    """Get registry statistics.
    
    Returns:
      Dictionary with registry statistics
    """
    total_agents = len(self._agents)
    status_counts = {}
    environment_counts = {}
    template_counts = {}
    
    for metadata in self._metadata.values():
      # Count by status
      status = metadata.status.value
      status_counts[status] = status_counts.get(status, 0) + 1
      
      # Count by environment
      env = metadata.environment
      environment_counts[env] = environment_counts.get(env, 0) + 1
      
      # Count by template
      template = metadata.template_id
      template_counts[template] = template_counts.get(template, 0) + 1
    
    return {
        'total_agents': total_agents,
        'status_distribution': status_counts,
        'environment_distribution': environment_counts,
        'template_distribution': template_counts,
        'total_capabilities': len(self._capability_index),
        'most_common_capabilities': self._get_most_common_capabilities()
    }
  
  def _extract_agent_capabilities(self, agent: BaseAgent, metadata: AgentMetadata) -> None:
    """Extract capabilities from an agent instance.
    
    Args:
      agent: Agent instance
      metadata: Metadata to update with capabilities
    """
    # Basic capabilities all agents have
    metadata.add_capability(
        "agent_execution",
        "Can execute agent logic",
        required=True
    )
    
    # Extract capabilities based on agent type
    from ..agents.llm_agent import LlmAgent
    from ..agents.sequential_agent import SequentialAgent
    from ..agents.parallel_agent import ParallelAgent
    from ..agents.loop_agent import LoopAgent
    
    if isinstance(agent, LlmAgent):
      metadata.add_capability(
          "llm_interaction",
          "Can interact with language models",
          {"model": getattr(agent, 'model', 'unknown')},
          required=True
      )
      
      # Tool capabilities
      if hasattr(agent, 'tools') and agent.tools:
        metadata.add_capability(
            "tool_usage",
            "Can use tools",
            {"tool_count": len(agent.tools)}
        )
      
      # Code execution capability
      if hasattr(agent, 'code_executor') and agent.code_executor:
        metadata.add_capability(
            "code_execution",
            "Can execute code"
        )
    
    elif isinstance(agent, SequentialAgent):
      metadata.add_capability(
          "sequential_orchestration",
          "Can orchestrate agents sequentially",
          {"sub_agent_count": len(agent.sub_agents)}
      )
    
    elif isinstance(agent, ParallelAgent):
      metadata.add_capability(
          "parallel_orchestration",
          "Can orchestrate agents in parallel",
          {"sub_agent_count": len(agent.sub_agents)}
      )
    
    elif isinstance(agent, LoopAgent):
      metadata.add_capability(
          "loop_orchestration",
          "Can orchestrate agents in loops",
          {
              "sub_agent_count": len(agent.sub_agents),
              "max_iterations": getattr(agent, 'max_iterations', None)
          }
      )
  
  def _update_capability_index(self, agent_id: str, metadata: AgentMetadata) -> None:
    """Update the capability index for an agent.
    
    Args:
      agent_id: Agent ID
      metadata: Agent metadata with capabilities
    """
    for capability in metadata.capabilities:
      if capability.name not in self._capability_index:
        self._capability_index[capability.name] = []
      
      if agent_id not in self._capability_index[capability.name]:
        self._capability_index[capability.name].append(agent_id)
  
  def _get_most_common_capabilities(self, limit: int = 5) -> List[Dict[str, Any]]:
    """Get the most common capabilities.
    
    Args:
      limit: Maximum number of capabilities to return
    
    Returns:
      List of capability information sorted by frequency
    """
    capability_counts = [
        {
            "name": capability,
            "agent_count": len(agent_ids)
        }
        for capability, agent_ids in self._capability_index.items()
    ]
    
    # Sort by agent count (descending)
    capability_counts.sort(key=lambda x: x["agent_count"], reverse=True)
    
    return capability_counts[:limit]

