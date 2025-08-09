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

"""Agent registry and management system for orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

# Import ADK components
try:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.tools.agent_tool import AgentTool
    from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
except ImportError:
    # Fallback for development/testing
    BaseAgent = object
    LlmAgent = object
    AgentTool = object
    AgentCardBuilder = object

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status of an agent in the registry."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentCapability(BaseModel):
    """Represents a capability that an agent provides."""
    
    name: str = Field(description="Name of the capability")
    description: str = Field(description="Description of what this capability does")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence level (0-1)")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class AgentRegistration(BaseModel):
    """Registration information for an agent."""
    
    agent_name: str = Field(description="Unique name of the agent")
    agent_type: str = Field(description="Type of agent (LlmAgent, SequentialAgent, etc.)")
    description: str = Field(description="Description of agent functionality")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Current status")
    registered_at: datetime = Field(default_factory=datetime.now, description="Registration timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    usage_count: int = Field(default=0, description="Number of times agent has been used")
    average_response_time: float = Field(default=0.0, description="Average response time in seconds")
    success_rate: float = Field(default=1.0, description="Success rate (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentHealthCheck(BaseModel):
    """Health check result for an agent."""
    
    agent_name: str = Field(description="Name of the agent")
    is_healthy: bool = Field(description="Whether the agent is healthy")
    response_time: float = Field(description="Response time for health check")
    error_message: Optional[str] = Field(default=None, description="Error message if unhealthy")
    checked_at: datetime = Field(default_factory=datetime.now, description="Health check timestamp")


class AgentRegistry:
    """Registry for managing agent instances and their metadata.
    
    Provides centralized management of agent lifecycle, capabilities,
    health monitoring, and discovery services.
    """
    
    def __init__(
        self,
        health_check_interval: int = 300,  # 5 minutes
        enable_auto_health_checks: bool = True,
    ):
        """Initialize the agent registry.
        
        Args:
            health_check_interval: Interval between health checks in seconds
            enable_auto_health_checks: Whether to enable automatic health checks
        """
        self.health_check_interval = health_check_interval
        self.enable_auto_health_checks = enable_auto_health_checks
        
        # Internal storage
        self._agents: Dict[str, BaseAgent] = {}
        self._registrations: Dict[str, AgentRegistration] = {}
        self._agent_tools: Dict[str, AgentTool] = {}
        self._health_status: Dict[str, AgentHealthCheck] = {}
        self._capability_index: Dict[str, Set[str]] = {}  # capability -> agent names
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("Initialized AgentRegistry")
        
        if self.enable_auto_health_checks:
            self._start_health_monitoring()
    
    def register_agent(
        self,
        agent: BaseAgent,
        capabilities: Optional[List[AgentCapability]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        create_tool: bool = True,
    ) -> AgentRegistration:
        """Register an agent with the registry.
        
        Args:
            agent: The agent instance to register
            capabilities: List of capabilities this agent provides
            metadata: Additional metadata for the agent
            create_tool: Whether to create an AgentTool wrapper
            
        Returns:
            Agent registration information
            
        Raises:
            ValueError: If agent name is invalid or already registered
        """
        if not hasattr(agent, 'name') or not agent.name:
            raise ValueError("Agent must have a valid name")
        
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered")
        
        # Extract capabilities if not provided
        if capabilities is None:
            capabilities = self._extract_agent_capabilities(agent)
        
        # Create registration
        registration = AgentRegistration(
            agent_name=agent.name,
            agent_type=agent.__class__.__name__,
            description=getattr(agent, 'description', f"Agent: {agent.name}"),
            capabilities=capabilities,
            metadata=metadata or {},
        )
        
        # Store agent and registration
        self._agents[agent.name] = agent
        self._registrations[agent.name] = registration
        
        # Update capability index
        for capability in capabilities:
            if capability.name not in self._capability_index:
                self._capability_index[capability.name] = set()
            self._capability_index[capability.name].add(agent.name)
        
        # Create agent tool if requested
        if create_tool:
            try:
                agent_tool = AgentTool(agent)
                self._agent_tools[agent.name] = agent_tool
                logger.debug(f"Created AgentTool for '{agent.name}'")
            except Exception as e:
                logger.warning(f"Failed to create AgentTool for '{agent.name}': {e}")
        
        logger.info(f"Registered agent: {agent.name} ({registration.agent_type})")
        return registration
    
    def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent from the registry.
        
        Args:
            agent_name: Name of the agent to unregister
            
        Returns:
            True if agent was successfully unregistered
        """
        if agent_name not in self._agents:
            return False
        
        # Remove from capability index
        registration = self._registrations.get(agent_name)
        if registration:
            for capability in registration.capabilities:
                if capability.name in self._capability_index:
                    self._capability_index[capability.name].discard(agent_name)
                    if not self._capability_index[capability.name]:
                        del self._capability_index[capability.name]
        
        # Remove from all storage
        del self._agents[agent_name]
        del self._registrations[agent_name]
        
        if agent_name in self._agent_tools:
            del self._agent_tools[agent_name]
        
        if agent_name in self._health_status:
            del self._health_status[agent_name]
        
        logger.info(f"Unregistered agent: {agent_name}")
        return True
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent instance or None if not found
        """
        return self._agents.get(agent_name)
    
    def get_agent_tool(self, agent_name: str) -> Optional[AgentTool]:
        """Get an agent tool by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentTool instance or None if not found
        """
        return self._agent_tools.get(agent_name)
    
    def get_registration(self, agent_name: str) -> Optional[AgentRegistration]:
        """Get agent registration information.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent registration or None if not found
        """
        return self._registrations.get(agent_name)
    
    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        capability: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> List[str]:
        """List registered agents with optional filtering.
        
        Args:
            status: Filter by agent status
            capability: Filter by capability name
            agent_type: Filter by agent type
            
        Returns:
            List of agent names matching the criteria
        """
        agents = []
        
        for name, registration in self._registrations.items():
            # Filter by status
            if status and registration.status != status:
                continue
            
            # Filter by capability
            if capability:
                has_capability = any(
                    cap.name == capability for cap in registration.capabilities
                )
                if not has_capability:
                    continue
            
            # Filter by agent type
            if agent_type and registration.agent_type != agent_type:
                continue
            
            agents.append(name)
        
        return agents
    
    def find_agents_by_capability(
        self,
        capability: str,
        min_confidence: float = 0.0,
    ) -> List[str]:
        """Find agents that provide a specific capability.
        
        Args:
            capability: Name of the capability to search for
            min_confidence: Minimum confidence level required
            
        Returns:
            List of agent names that provide the capability
        """
        if capability not in self._capability_index:
            return []
        
        matching_agents = []
        for agent_name in self._capability_index[capability]:
            registration = self._registrations.get(agent_name)
            if registration:
                for cap in registration.capabilities:
                    if cap.name == capability and cap.confidence >= min_confidence:
                        matching_agents.append(agent_name)
                        break
        
        # Sort by confidence (highest first)
        def get_confidence(agent_name: str) -> float:
            registration = self._registrations.get(agent_name)
            if registration:
                for cap in registration.capabilities:
                    if cap.name == capability:
                        return cap.confidence
            return 0.0
        
        matching_agents.sort(key=get_confidence, reverse=True)
        return matching_agents
    
    def get_all_capabilities(self) -> List[str]:
        """Get list of all available capabilities.
        
        Returns:
            List of capability names
        """
        return list(self._capability_index.keys())
    
    def update_agent_status(
        self,
        agent_name: str,
        status: AgentStatus,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an agent's status.
        
        Args:
            agent_name: Name of the agent
            status: New status
            metadata: Optional metadata to update
            
        Returns:
            True if status was updated successfully
        """
        if agent_name not in self._registrations:
            return False
        
        registration = self._registrations[agent_name]
        registration.status = status
        
        if metadata:
            registration.metadata.update(metadata)
        
        logger.debug(f"Updated status for '{agent_name}': {status}")
        return True
    
    def record_agent_usage(
        self,
        agent_name: str,
        response_time: float,
        success: bool,
    ) -> None:
        """Record usage statistics for an agent.
        
        Args:
            agent_name: Name of the agent
            response_time: Response time in seconds
            success: Whether the operation was successful
        """
        if agent_name not in self._registrations:
            return
        
        registration = self._registrations[agent_name]
        registration.last_used = datetime.now()
        registration.usage_count += 1
        
        # Update average response time
        if registration.usage_count == 1:
            registration.average_response_time = response_time
        else:
            current_avg = registration.average_response_time
            count = registration.usage_count
            registration.average_response_time = (
                (current_avg * (count - 1) + response_time) / count
            )
        
        # Update success rate
        if registration.usage_count == 1:
            registration.success_rate = 1.0 if success else 0.0
        else:
            current_rate = registration.success_rate
            count = registration.usage_count
            success_count = int(current_rate * (count - 1)) + (1 if success else 0)
            registration.success_rate = success_count / count
    
    async def health_check_agent(self, agent_name: str) -> AgentHealthCheck:
        """Perform a health check on a specific agent.
        
        Args:
            agent_name: Name of the agent to check
            
        Returns:
            Health check result
        """
        agent = self.get_agent(agent_name)
        if not agent:
            return AgentHealthCheck(
                agent_name=agent_name,
                is_healthy=False,
                response_time=0.0,
                error_message="Agent not found",
            )
        
        start_time = time.time()
        try:
            # Perform a simple health check
            # In a real implementation, this might call a health check method on the agent
            await asyncio.sleep(0.01)  # Simulate health check
            
            response_time = time.time() - start_time
            health_check = AgentHealthCheck(
                agent_name=agent_name,
                is_healthy=True,
                response_time=response_time,
            )
            
            # Update agent status based on health
            self.update_agent_status(agent_name, AgentStatus.ACTIVE)
            
        except Exception as e:
            response_time = time.time() - start_time
            health_check = AgentHealthCheck(
                agent_name=agent_name,
                is_healthy=False,
                response_time=response_time,
                error_message=str(e),
            )
            
            # Update agent status to error
            self.update_agent_status(agent_name, AgentStatus.ERROR)
        
        self._health_status[agent_name] = health_check
        return health_check
    
    async def health_check_all_agents(self) -> Dict[str, AgentHealthCheck]:
        """Perform health checks on all registered agents.
        
        Returns:
            Dictionary mapping agent names to health check results
        """
        tasks = []
        for agent_name in self._agents.keys():
            task = self.health_check_agent(agent_name)
            tasks.append((agent_name, task))
        
        results = {}
        if tasks:
            health_checks = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for (agent_name, _), health_check in zip(tasks, health_checks):
                if isinstance(health_check, Exception):
                    results[agent_name] = AgentHealthCheck(
                        agent_name=agent_name,
                        is_healthy=False,
                        response_time=0.0,
                        error_message=str(health_check),
                    )
                else:
                    results[agent_name] = health_check
        
        return results
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics.
        
        Returns:
            Dictionary containing registry statistics
        """
        total_agents = len(self._agents)
        status_counts = {}
        
        for registration in self._registrations.values():
            status = registration.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_usage = sum(reg.usage_count for reg in self._registrations.values())
        avg_success_rate = (
            sum(reg.success_rate for reg in self._registrations.values()) / total_agents
            if total_agents > 0 else 0.0
        )
        
        return {
            'total_agents': total_agents,
            'status_distribution': status_counts,
            'total_capabilities': len(self._capability_index),
            'total_usage_count': total_usage,
            'average_success_rate': avg_success_rate,
            'health_checks_performed': len(self._health_status),
        }
    
    def _extract_agent_capabilities(self, agent: BaseAgent) -> List[AgentCapability]:
        """Extract capabilities from an agent instance.
        
        Args:
            agent: The agent to analyze
            
        Returns:
            List of detected capabilities
        """
        capabilities = []
        
        # Basic capability based on agent type
        agent_type = agent.__class__.__name__
        if agent_type == "LlmAgent":
            capabilities.append(AgentCapability(
                name="language_processing",
                description="Can process and generate natural language",
                confidence=0.9,
                tags=["llm", "text"],
            ))
        
        # Try to extract capabilities from agent description
        description = getattr(agent, 'description', '').lower()
        if 'search' in description:
            capabilities.append(AgentCapability(
                name="search",
                description="Can perform search operations",
                confidence=0.8,
                tags=["search", "retrieval"],
            ))
        
        if 'analyze' in description or 'analysis' in description:
            capabilities.append(AgentCapability(
                name="analysis",
                description="Can perform data analysis",
                confidence=0.8,
                tags=["analysis", "data"],
            ))
        
        # Default capability if none detected
        if not capabilities:
            capabilities.append(AgentCapability(
                name="general",
                description="General purpose agent",
                confidence=0.5,
                tags=["general"],
            ))
        
        return capabilities
    
    def _start_health_monitoring(self) -> None:
        """Start the background health monitoring task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
            logger.info("Started health monitoring")
    
    async def _health_monitoring_loop(self) -> None:
        """Background loop for periodic health checks."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.health_check_interval
                )
                break  # Shutdown event was set
            except asyncio.TimeoutError:
                # Perform health checks
                try:
                    await self.health_check_all_agents()
                    logger.debug("Completed periodic health checks")
                except Exception as e:
                    logger.error(f"Error during health checks: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the registry and cleanup resources."""
        logger.info("Shutting down AgentRegistry")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for health monitoring task to complete
        if self._health_check_task and not self._health_check_task.done():
            try:
                await asyncio.wait_for(self._health_check_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._health_check_task.cancel()
        
        # Clear all data
        self._agents.clear()
        self._registrations.clear()
        self._agent_tools.clear()
        self._health_status.clear()
        self._capability_index.clear()
        
        logger.info("AgentRegistry shutdown complete")

