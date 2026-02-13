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

"""Agent manager for lifecycle and resource management."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .agent_registry import AgentRegistry, AgentStatus, AgentRegistration

# Import ADK components
try:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.agents.sequential_agent import SequentialAgent
    from google.adk.agents.parallel_agent import ParallelAgent
    from google.adk.agents.loop_agent import LoopAgent
except ImportError:
    # Fallback for development/testing
    BaseAgent = object
    LlmAgent = object
    SequentialAgent = object
    ParallelAgent = object
    LoopAgent = object

logger = logging.getLogger(__name__)


class AgentLoadBalancer:
    """Load balancer for distributing requests across multiple agents."""
    
    def __init__(self, registry: AgentRegistry):
        """Initialize the load balancer.
        
        Args:
            registry: Agent registry instance
        """
        self.registry = registry
        self._request_counts: Dict[str, int] = {}
    
    def select_agent(
        self,
        capability: str,
        exclude_agents: Optional[Set[str]] = None,
        strategy: str = "round_robin",
    ) -> Optional[str]:
        """Select the best agent for a given capability.
        
        Args:
            capability: Required capability
            exclude_agents: Set of agent names to exclude
            strategy: Load balancing strategy (round_robin, least_used, best_performance)
            
        Returns:
            Selected agent name or None if no suitable agent found
        """
        candidates = self.registry.find_agents_by_capability(capability)
        
        # Filter out excluded agents and inactive agents
        available_agents = []
        for agent_name in candidates:
            if exclude_agents and agent_name in exclude_agents:
                continue
            
            registration = self.registry.get_registration(agent_name)
            if registration and registration.status == AgentStatus.ACTIVE:
                available_agents.append(agent_name)
        
        if not available_agents:
            return None
        
        if strategy == "round_robin":
            return self._round_robin_selection(available_agents)
        elif strategy == "least_used":
            return self._least_used_selection(available_agents)
        elif strategy == "best_performance":
            return self._best_performance_selection(available_agents)
        else:
            return available_agents[0]  # Default to first available
    
    def _round_robin_selection(self, agents: List[str]) -> str:
        """Select agent using round-robin strategy."""
        # Simple round-robin based on request counts
        min_requests = min(self._request_counts.get(agent, 0) for agent in agents)
        for agent in agents:
            if self._request_counts.get(agent, 0) == min_requests:
                self._request_counts[agent] = self._request_counts.get(agent, 0) + 1
                return agent
        return agents[0]
    
    def _least_used_selection(self, agents: List[str]) -> str:
        """Select the least used agent."""
        def get_usage_count(agent_name: str) -> int:
            registration = self.registry.get_registration(agent_name)
            return registration.usage_count if registration else 0
        
        return min(agents, key=get_usage_count)
    
    def _best_performance_selection(self, agents: List[str]) -> str:
        """Select agent with best performance metrics."""
        def get_performance_score(agent_name: str) -> float:
            registration = self.registry.get_registration(agent_name)
            if not registration:
                return 0.0
            
            # Combine success rate and response time (lower is better for response time)
            success_weight = 0.7
            speed_weight = 0.3
            
            success_score = registration.success_rate
            # Normalize response time (assume 1 second is baseline)
            speed_score = max(0.0, 1.0 - (registration.average_response_time / 1.0))
            
            return success_weight * success_score + speed_weight * speed_score
        
        return max(agents, key=get_performance_score)


class AgentPool:
    """Pool of agents for a specific capability or use case."""
    
    def __init__(
        self,
        name: str,
        capability: str,
        min_agents: int = 1,
        max_agents: int = 5,
        scale_threshold: float = 0.8,
    ):
        """Initialize the agent pool.
        
        Args:
            name: Name of the pool
            capability: Capability this pool provides
            min_agents: Minimum number of agents to maintain
            max_agents: Maximum number of agents allowed
            scale_threshold: Utilization threshold for scaling up
        """
        self.name = name
        self.capability = capability
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.scale_threshold = scale_threshold
        
        self._agents: Set[str] = set()
        self._busy_agents: Set[str] = set()
        self._pending_requests: int = 0
    
    def add_agent(self, agent_name: str) -> None:
        """Add an agent to the pool."""
        self._agents.add(agent_name)
        logger.debug(f"Added agent '{agent_name}' to pool '{self.name}'")
    
    def remove_agent(self, agent_name: str) -> bool:
        """Remove an agent from the pool."""
        if agent_name in self._agents:
            self._agents.remove(agent_name)
            self._busy_agents.discard(agent_name)
            logger.debug(f"Removed agent '{agent_name}' from pool '{self.name}'")
            return True
        return False
    
    def get_available_agent(self) -> Optional[str]:
        """Get an available agent from the pool."""
        available = self._agents - self._busy_agents
        return next(iter(available)) if available else None
    
    def mark_agent_busy(self, agent_name: str) -> None:
        """Mark an agent as busy."""
        if agent_name in self._agents:
            self._busy_agents.add(agent_name)
    
    def mark_agent_available(self, agent_name: str) -> None:
        """Mark an agent as available."""
        self._busy_agents.discard(agent_name)
    
    def get_utilization(self) -> float:
        """Get current pool utilization (0.0 to 1.0)."""
        if not self._agents:
            return 0.0
        return len(self._busy_agents) / len(self._agents)
    
    def should_scale_up(self) -> bool:
        """Check if the pool should scale up."""
        return (
            len(self._agents) < self.max_agents and
            self.get_utilization() >= self.scale_threshold
        )
    
    def should_scale_down(self) -> bool:
        """Check if the pool should scale down."""
        return (
            len(self._agents) > self.min_agents and
            self.get_utilization() < (self.scale_threshold * 0.5)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'name': self.name,
            'capability': self.capability,
            'total_agents': len(self._agents),
            'busy_agents': len(self._busy_agents),
            'available_agents': len(self._agents) - len(self._busy_agents),
            'utilization': self.get_utilization(),
            'pending_requests': self._pending_requests,
        }


class AgentManager:
    """Comprehensive agent lifecycle and resource management.
    
    Provides advanced features including:
    - Agent pooling and load balancing
    - Automatic scaling based on demand
    - Resource monitoring and optimization
    - Agent lifecycle management
    """
    
    def __init__(
        self,
        registry: AgentRegistry,
        enable_auto_scaling: bool = True,
        scaling_check_interval: int = 60,
    ):
        """Initialize the agent manager.
        
        Args:
            registry: Agent registry instance
            enable_auto_scaling: Whether to enable automatic scaling
            scaling_check_interval: Interval for scaling checks in seconds
        """
        self.registry = registry
        self.enable_auto_scaling = enable_auto_scaling
        self.scaling_check_interval = scaling_check_interval
        
        self.load_balancer = AgentLoadBalancer(registry)
        self._pools: Dict[str, AgentPool] = {}
        self._scaling_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("Initialized AgentManager")
        
        if self.enable_auto_scaling:
            self._start_auto_scaling()
    
    def create_agent_pool(
        self,
        name: str,
        capability: str,
        min_agents: int = 1,
        max_agents: int = 5,
        scale_threshold: float = 0.8,
    ) -> AgentPool:
        """Create a new agent pool.
        
        Args:
            name: Name of the pool
            capability: Capability this pool provides
            min_agents: Minimum number of agents
            max_agents: Maximum number of agents
            scale_threshold: Utilization threshold for scaling
            
        Returns:
            Created agent pool
        """
        if name in self._pools:
            raise ValueError(f"Pool '{name}' already exists")
        
        pool = AgentPool(name, capability, min_agents, max_agents, scale_threshold)
        self._pools[name] = pool
        
        # Add existing agents with this capability to the pool
        capable_agents = self.registry.find_agents_by_capability(capability)
        for agent_name in capable_agents[:max_agents]:
            pool.add_agent(agent_name)
        
        logger.info(f"Created agent pool '{name}' for capability '{capability}'")
        return pool
    
    def get_agent_pool(self, name: str) -> Optional[AgentPool]:
        """Get an agent pool by name."""
        return self._pools.get(name)
    
    def remove_agent_pool(self, name: str) -> bool:
        """Remove an agent pool."""
        if name in self._pools:
            del self._pools[name]
            logger.info(f"Removed agent pool '{name}'")
            return True
        return False
    
    async def get_agent_for_capability(
        self,
        capability: str,
        pool_name: Optional[str] = None,
        strategy: str = "round_robin",
    ) -> Optional[str]:
        """Get an agent for a specific capability.
        
        Args:
            capability: Required capability
            pool_name: Optional pool name to use
            strategy: Load balancing strategy
            
        Returns:
            Selected agent name or None if no suitable agent found
        """
        if pool_name and pool_name in self._pools:
            # Use pool-based selection
            pool = self._pools[pool_name]
            agent_name = pool.get_available_agent()
            if agent_name:
                pool.mark_agent_busy(agent_name)
                return agent_name
        
        # Use load balancer
        return self.load_balancer.select_agent(capability, strategy=strategy)
    
    def release_agent(self, agent_name: str, pool_name: Optional[str] = None) -> None:
        """Release an agent back to the available pool.
        
        Args:
            agent_name: Name of the agent to release
            pool_name: Optional pool name
        """
        if pool_name and pool_name in self._pools:
            pool = self._pools[pool_name]
            pool.mark_agent_available(agent_name)
        
        # Update load balancer request counts
        if agent_name in self.load_balancer._request_counts:
            self.load_balancer._request_counts[agent_name] -= 1
    
    async def scale_pool(self, pool_name: str, target_size: int) -> bool:
        """Scale a pool to a target size.
        
        Args:
            pool_name: Name of the pool to scale
            target_size: Target number of agents
            
        Returns:
            True if scaling was successful
        """
        if pool_name not in self._pools:
            return False
        
        pool = self._pools[pool_name]
        current_size = len(pool._agents)
        
        if target_size > current_size:
            # Scale up - find more agents with the capability
            needed = target_size - current_size
            available_agents = self.registry.find_agents_by_capability(pool.capability)
            
            # Filter out agents already in the pool
            candidates = [
                agent for agent in available_agents
                if agent not in pool._agents
            ]
            
            for agent_name in candidates[:needed]:
                pool.add_agent(agent_name)
                logger.info(f"Scaled up pool '{pool_name}': added '{agent_name}'")
        
        elif target_size < current_size:
            # Scale down - remove least used agents
            to_remove = current_size - target_size
            
            # Get agents sorted by usage (least used first)
            agents_by_usage = []
            for agent_name in pool._agents:
                if agent_name not in pool._busy_agents:  # Don't remove busy agents
                    registration = self.registry.get_registration(agent_name)
                    usage = registration.usage_count if registration else 0
                    agents_by_usage.append((usage, agent_name))
            
            agents_by_usage.sort()  # Sort by usage (ascending)
            
            for _, agent_name in agents_by_usage[:to_remove]:
                pool.remove_agent(agent_name)
                logger.info(f"Scaled down pool '{pool_name}': removed '{agent_name}'")
        
        return True
    
    async def auto_scale_pools(self) -> None:
        """Automatically scale pools based on utilization."""
        for pool_name, pool in self._pools.items():
            try:
                if pool.should_scale_up():
                    new_size = min(len(pool._agents) + 1, pool.max_agents)
                    await self.scale_pool(pool_name, new_size)
                    logger.info(f"Auto-scaled up pool '{pool_name}' to {new_size} agents")
                
                elif pool.should_scale_down():
                    new_size = max(len(pool._agents) - 1, pool.min_agents)
                    await self.scale_pool(pool_name, new_size)
                    logger.info(f"Auto-scaled down pool '{pool_name}' to {new_size} agents")
            
            except Exception as e:
                logger.error(f"Error auto-scaling pool '{pool_name}': {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        pool_stats = {}
        for pool_name, pool in self._pools.items():
            pool_stats[pool_name] = pool.get_stats()
        
        total_agents = len(self.registry.list_agents())
        total_busy = sum(len(pool._busy_agents) for pool in self._pools.values())
        
        return {
            'total_agents_managed': total_agents,
            'total_busy_agents': total_busy,
            'total_pools': len(self._pools),
            'pools': pool_stats,
            'load_balancer_requests': dict(self.load_balancer._request_counts),
            'auto_scaling_enabled': self.enable_auto_scaling,
        }
    
    def optimize_agent_distribution(self) -> Dict[str, Any]:
        """Optimize agent distribution across pools.
        
        Returns:
            Optimization report
        """
        optimization_report = {
            'recommendations': [],
            'current_distribution': {},
            'optimal_distribution': {},
        }
        
        # Analyze current distribution
        for pool_name, pool in self._pools.items():
            stats = pool.get_stats()
            optimization_report['current_distribution'][pool_name] = stats
            
            # Generate recommendations
            if pool.get_utilization() > 0.9:
                optimization_report['recommendations'].append({
                    'pool': pool_name,
                    'action': 'scale_up',
                    'reason': 'High utilization detected',
                    'current_size': len(pool._agents),
                    'recommended_size': min(len(pool._agents) + 1, pool.max_agents),
                })
            
            elif pool.get_utilization() < 0.2 and len(pool._agents) > pool.min_agents:
                optimization_report['recommendations'].append({
                    'pool': pool_name,
                    'action': 'scale_down',
                    'reason': 'Low utilization detected',
                    'current_size': len(pool._agents),
                    'recommended_size': max(len(pool._agents) - 1, pool.min_agents),
                })
        
        return optimization_report
    
    def _start_auto_scaling(self) -> None:
        """Start the auto-scaling background task."""
        if self._scaling_task is None or self._scaling_task.done():
            self._scaling_task = asyncio.create_task(self._auto_scaling_loop())
            logger.info("Started auto-scaling")
    
    async def _auto_scaling_loop(self) -> None:
        """Background loop for automatic scaling."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.scaling_check_interval
                )
                break  # Shutdown event was set
            except asyncio.TimeoutError:
                # Perform auto-scaling
                try:
                    await self.auto_scale_pools()
                    logger.debug("Completed auto-scaling check")
                except Exception as e:
                    logger.error(f"Error during auto-scaling: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the agent manager."""
        logger.info("Shutting down AgentManager")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for scaling task to complete
        if self._scaling_task and not self._scaling_task.done():
            try:
                await asyncio.wait_for(self._scaling_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._scaling_task.cancel()
        
        # Clear pools
        self._pools.clear()
        
        logger.info("AgentManager shutdown complete")

