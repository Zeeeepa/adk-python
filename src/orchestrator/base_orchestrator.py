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

"""Base orchestrator class providing core orchestration capabilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from google.genai import types
from pydantic import BaseModel, Field

# Import ADK components (these would be the actual imports in a real implementation)
try:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.tools.agent_tool import AgentTool
except ImportError:
    # Fallback for development/testing
    BaseAgent = object
    LlmAgent = object
    InvocationContext = object
    CallbackContext = object
    AgentTool = object

logger = logging.getLogger(__name__)


class OrchestratorConfig(BaseModel):
    """Configuration for orchestrator behavior."""
    
    max_delegation_depth: int = Field(default=5, description="Maximum delegation depth")
    enable_parallel_execution: bool = Field(default=True, description="Enable parallel agent execution")
    enable_state_persistence: bool = Field(default=True, description="Enable state persistence")
    timeout_seconds: int = Field(default=300, description="Default timeout for agent execution")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed operations")


class OrchestratorMetrics(BaseModel):
    """Metrics tracking for orchestrator operations."""
    
    total_requests: int = Field(default=0)
    successful_delegations: int = Field(default=0)
    failed_delegations: int = Field(default=0)
    average_response_time: float = Field(default=0.0)
    active_agents: int = Field(default=0)


class BaseOrchestrator(ABC):
    """Base class for all orchestrator implementations.
    
    Provides core functionality for agent orchestration including:
    - Agent lifecycle management
    - Request routing and delegation
    - State management and persistence
    - Error handling and recovery
    - Metrics collection and monitoring
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """Initialize the base orchestrator.
        
        Args:
            name: Unique name for this orchestrator instance
            description: Optional description of orchestrator capabilities
            config: Configuration settings for orchestrator behavior
        """
        self.name = name
        self.description = description or f"Base orchestrator: {name}"
        self.config = config or OrchestratorConfig()
        self.metrics = OrchestratorMetrics()
        
        # Internal state
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_tools: Dict[str, AgentTool] = {}
        self._active_contexts: Dict[str, InvocationContext] = {}
        self._state_store: Dict[str, Any] = {}
        
        logger.info(f"Initialized orchestrator: {self.name}")
    
    @abstractmethod
    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process an incoming request through the orchestration system.
        
        Args:
            request: The user request to process
            context: Optional context information
            
        Returns:
            Dictionary containing the response and metadata
        """
        pass
    
    def register_agent(
        self,
        agent: BaseAgent,
        capabilities: Optional[List[str]] = None,
        as_tool: bool = True,
    ) -> None:
        """Register an agent with the orchestrator.
        
        Args:
            agent: The agent instance to register
            capabilities: List of capabilities this agent provides
            as_tool: Whether to also register the agent as a tool
        """
        if not hasattr(agent, 'name') or not agent.name:
            raise ValueError("Agent must have a valid name")
            
        self._agents[agent.name] = agent
        
        if as_tool:
            try:
                agent_tool = AgentTool(agent)
                self._agent_tools[agent.name] = agent_tool
                logger.info(f"Registered agent '{agent.name}' as tool")
            except Exception as e:
                logger.warning(f"Failed to register agent '{agent.name}' as tool: {e}")
        
        logger.info(f"Registered agent: {agent.name}")
        self.metrics.active_agents += 1
    
    def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent from the orchestrator.
        
        Args:
            agent_name: Name of the agent to unregister
            
        Returns:
            True if agent was successfully unregistered
        """
        removed = False
        
        if agent_name in self._agents:
            del self._agents[agent_name]
            removed = True
            self.metrics.active_agents -= 1
            
        if agent_name in self._agent_tools:
            del self._agent_tools[agent_name]
            
        if removed:
            logger.info(f"Unregistered agent: {agent_name}")
            
        return removed
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get a registered agent by name.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            The agent instance or None if not found
        """
        return self._agents.get(agent_name)
    
    def get_agent_tool(self, agent_name: str) -> Optional[AgentTool]:
        """Get an agent tool by name.
        
        Args:
            agent_name: Name of the agent tool to retrieve
            
        Returns:
            The agent tool instance or None if not found
        """
        return self._agent_tools.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """Get list of all registered agent names.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def get_metrics(self) -> OrchestratorMetrics:
        """Get current orchestrator metrics.
        
        Returns:
            Current metrics snapshot
        """
        return self.metrics.model_copy()
    
    def store_state(self, key: str, value: Any) -> None:
        """Store state information.
        
        Args:
            key: State key
            value: State value
        """
        self._state_store[key] = value
        logger.debug(f"Stored state: {key}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieve state information.
        
        Args:
            key: State key
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        return self._state_store.get(key, default)
    
    def clear_state(self, key: Optional[str] = None) -> None:
        """Clear state information.
        
        Args:
            key: Specific key to clear, or None to clear all state
        """
        if key is None:
            self._state_store.clear()
            logger.info("Cleared all state")
        elif key in self._state_store:
            del self._state_store[key]
            logger.debug(f"Cleared state: {key}")
    
    async def _create_invocation_context(
        self,
        request: str,
        agent: BaseAgent,
        parent_context: Optional[InvocationContext] = None,
    ) -> InvocationContext:
        """Create an invocation context for agent execution.
        
        Args:
            request: The request being processed
            agent: The agent that will process the request
            parent_context: Optional parent context
            
        Returns:
            New invocation context
        """
        # This would create a proper InvocationContext in the real implementation
        # For now, return a mock context
        context_data = {
            'request': request,
            'agent_name': agent.name if hasattr(agent, 'name') else 'unknown',
            'orchestrator': self.name,
            'parent_context': parent_context,
        }
        
        # Store context for tracking
        context_id = f"{self.name}_{len(self._active_contexts)}"
        self._active_contexts[context_id] = context_data
        
        return context_data
    
    def _cleanup_context(self, context_id: str) -> None:
        """Clean up an invocation context.
        
        Args:
            context_id: ID of the context to clean up
        """
        if context_id in self._active_contexts:
            del self._active_contexts[context_id]
            logger.debug(f"Cleaned up context: {context_id}")

