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

"""Hierarchical delegation engine for intelligent task routing."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from .agent_registry import AgentRegistry
from .agent_manager import AgentManager

# Import ADK components
try:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.invocation_context import InvocationContext
except ImportError:
    # Fallback for development/testing
    BaseAgent = object
    InvocationContext = object

logger = logging.getLogger(__name__)


class DelegationStrategy(str, Enum):
    """Strategy for delegation decisions."""
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCED = "load_balanced"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


class DelegationPriority(str, Enum):
    """Priority levels for delegation requests."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class DelegationRequest(BaseModel):
    """Request for agent delegation."""
    
    request_id: str = Field(description="Unique request identifier")
    original_request: str = Field(description="Original user request")
    required_capabilities: List[str] = Field(description="Required agent capabilities")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request context")
    priority: DelegationPriority = Field(default=DelegationPriority.NORMAL, description="Request priority")
    max_delegation_depth: int = Field(default=3, description="Maximum delegation depth")
    timeout_seconds: int = Field(default=300, description="Request timeout")
    exclude_agents: Set[str] = Field(default_factory=set, description="Agents to exclude")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DelegationResult(BaseModel):
    """Result of a delegation operation."""
    
    request_id: str = Field(description="Request identifier")
    success: bool = Field(description="Whether delegation was successful")
    selected_agent: Optional[str] = Field(default=None, description="Selected agent name")
    delegation_path: List[str] = Field(default_factory=list, description="Path of delegation")
    result: Any = Field(default=None, description="Delegation result")
    execution_time: float = Field(description="Execution time in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")


class DelegationDecision(BaseModel):
    """Decision made by the delegation engine."""
    
    agent_name: str = Field(description="Selected agent name")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in decision")
    reasoning: str = Field(description="Reasoning for the decision")
    alternative_agents: List[str] = Field(default_factory=list, description="Alternative agents")
    estimated_execution_time: float = Field(description="Estimated execution time")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")


class RequestAnalyzer:
    """Analyzes requests to determine delegation requirements."""
    
    def __init__(self, registry: AgentRegistry):
        """Initialize the request analyzer.
        
        Args:
            registry: Agent registry for capability lookup
        """
        self.registry = registry
        self._capability_keywords = {
            'search': ['search', 'find', 'lookup', 'query', 'retrieve'],
            'analysis': ['analyze', 'examine', 'study', 'evaluate', 'assess'],
            'generation': ['generate', 'create', 'write', 'produce', 'compose'],
            'processing': ['process', 'transform', 'convert', 'parse', 'extract'],
            'calculation': ['calculate', 'compute', 'sum', 'count', 'measure'],
            'communication': ['send', 'notify', 'message', 'email', 'alert'],
        }
    
    def analyze_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Analyze a request to identify required capabilities.
        
        Args:
            request: The request text to analyze
            context: Optional context information
            
        Returns:
            Tuple of (required_capabilities, confidence_scores)
        """
        request_lower = request.lower()
        context = context or {}
        
        detected_capabilities = {}
        
        # Keyword-based capability detection
        for capability, keywords in self._capability_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in request_lower:
                    score += 1.0
            
            if score > 0:
                # Normalize score based on number of keywords
                normalized_score = min(score / len(keywords), 1.0)
                detected_capabilities[capability] = normalized_score
        
        # Context-based capability enhancement
        if context.get('domain') == 'research':
            detected_capabilities['search'] = detected_capabilities.get('search', 0.0) + 0.3
            detected_capabilities['analysis'] = detected_capabilities.get('analysis', 0.0) + 0.2
        
        if context.get('output_format') in ['document', 'report']:
            detected_capabilities['generation'] = detected_capabilities.get('generation', 0.0) + 0.3
        
        # Filter capabilities with minimum confidence
        min_confidence = 0.1
        filtered_capabilities = {
            cap: score for cap, score in detected_capabilities.items()
            if score >= min_confidence
        }
        
        # Default to general capability if none detected
        if not filtered_capabilities:
            filtered_capabilities['general'] = 0.5
        
        required_capabilities = list(filtered_capabilities.keys())
        confidence_scores = filtered_capabilities
        
        logger.debug(f"Analyzed request capabilities: {required_capabilities}")
        return required_capabilities, confidence_scores
    
    def estimate_complexity(
        self,
        request: str,
        capabilities: List[str],
    ) -> Tuple[str, float]:
        """Estimate the complexity of a request.
        
        Args:
            request: The request text
            capabilities: Required capabilities
            
        Returns:
            Tuple of (complexity_level, estimated_time)
        """
        complexity_indicators = {
            'simple': ['simple', 'quick', 'basic', 'single'],
            'moderate': ['analyze', 'compare', 'process', 'multiple'],
            'complex': ['comprehensive', 'detailed', 'complex', 'advanced', 'research'],
        }
        
        request_lower = request.lower()
        complexity_scores = {'simple': 0, 'moderate': 0, 'complex': 0}
        
        # Keyword-based complexity detection
        for level, keywords in complexity_indicators.items():
            for keyword in keywords:
                if keyword in request_lower:
                    complexity_scores[level] += 1
        
        # Capability-based complexity adjustment
        if len(capabilities) > 2:
            complexity_scores['complex'] += 1
        elif len(capabilities) > 1:
            complexity_scores['moderate'] += 1
        
        # Request length as complexity indicator
        word_count = len(request.split())
        if word_count > 50:
            complexity_scores['complex'] += 1
        elif word_count > 20:
            complexity_scores['moderate'] += 1
        
        # Determine final complexity
        max_score = max(complexity_scores.values())
        if max_score == 0:
            complexity_level = 'simple'
        else:
            complexity_level = max(complexity_scores, key=complexity_scores.get)
        
        # Estimate execution time based on complexity
        time_estimates = {
            'simple': 30.0,    # 30 seconds
            'moderate': 120.0,  # 2 minutes
            'complex': 300.0,   # 5 minutes
        }
        
        estimated_time = time_estimates[complexity_level]
        
        return complexity_level, estimated_time


class DelegationEngine:
    """Intelligent delegation engine for hierarchical task routing.
    
    Provides sophisticated delegation capabilities including:
    - Multi-strategy agent selection
    - Performance-based routing
    - Load balancing and optimization
    - Hierarchical delegation with depth control
    - Risk assessment and fallback handling
    """
    
    def __init__(
        self,
        registry: AgentRegistry,
        manager: AgentManager,
        default_strategy: DelegationStrategy = DelegationStrategy.HYBRID,
        max_concurrent_delegations: int = 10,
    ):
        """Initialize the delegation engine.
        
        Args:
            registry: Agent registry for capability lookup
            manager: Agent manager for load balancing
            default_strategy: Default delegation strategy
            max_concurrent_delegations: Maximum concurrent delegations
        """
        self.registry = registry
        self.manager = manager
        self.default_strategy = default_strategy
        self.max_concurrent_delegations = max_concurrent_delegations
        
        self.request_analyzer = RequestAnalyzer(registry)
        
        # Internal state
        self._active_delegations: Dict[str, DelegationRequest] = {}
        self._delegation_history: List[DelegationResult] = []
        self._performance_cache: Dict[str, Dict[str, float]] = {}
        
        # Concurrency control
        self._delegation_semaphore = asyncio.Semaphore(max_concurrent_delegations)
        
        logger.info(f"Initialized DelegationEngine with strategy: {default_strategy}")
    
    async def delegate_request(
        self,
        request: DelegationRequest,
        strategy: Optional[DelegationStrategy] = None,
    ) -> DelegationResult:
        """Delegate a request to the most appropriate agent.
        
        Args:
            request: Delegation request
            strategy: Optional strategy override
            
        Returns:
            Delegation result
        """
        strategy = strategy or self.default_strategy
        start_time = asyncio.get_event_loop().time()
        
        # Check concurrency limits
        async with self._delegation_semaphore:
            try:
                # Store active delegation
                self._active_delegations[request.request_id] = request
                
                logger.info(f"Delegating request {request.request_id} with strategy: {strategy}")
                
                # Make delegation decision
                decision = await self._make_delegation_decision(request, strategy)
                
                if not decision:
                    return DelegationResult(
                        request_id=request.request_id,
                        success=False,
                        execution_time=asyncio.get_event_loop().time() - start_time,
                        error_message="No suitable agent found for delegation",
                    )
                
                # Execute delegation
                result = await self._execute_delegation(request, decision)
                
                # Update performance cache
                self._update_performance_cache(decision.agent_name, result)
                
                # Store in history
                self._delegation_history.append(result)
                
                # Cleanup
                self._active_delegations.pop(request.request_id, None)
                
                logger.info(f"Completed delegation {request.request_id} in {result.execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = asyncio.get_event_loop().time() - start_time
                error_result = DelegationResult(
                    request_id=request.request_id,
                    success=False,
                    execution_time=execution_time,
                    error_message=str(e),
                )
                
                self._delegation_history.append(error_result)
                self._active_delegations.pop(request.request_id, None)
                
                logger.error(f"Delegation {request.request_id} failed: {e}")
                return error_result
    
    async def _make_delegation_decision(
        self,
        request: DelegationRequest,
        strategy: DelegationStrategy,
    ) -> Optional[DelegationDecision]:
        """Make a delegation decision based on the specified strategy.
        
        Args:
            request: Delegation request
            strategy: Delegation strategy to use
            
        Returns:
            Delegation decision or None if no suitable agent found
        """
        if strategy == DelegationStrategy.CAPABILITY_BASED:
            return await self._capability_based_decision(request)
        elif strategy == DelegationStrategy.PERFORMANCE_BASED:
            return await self._performance_based_decision(request)
        elif strategy == DelegationStrategy.LOAD_BALANCED:
            return await self._load_balanced_decision(request)
        elif strategy == DelegationStrategy.HIERARCHICAL:
            return await self._hierarchical_decision(request)
        elif strategy == DelegationStrategy.HYBRID:
            return await self._hybrid_decision(request)
        else:
            return await self._capability_based_decision(request)
    
    async def _capability_based_decision(
        self,
        request: DelegationRequest,
    ) -> Optional[DelegationDecision]:
        """Make decision based purely on capability matching."""
        best_agent = None
        best_score = 0.0
        
        for capability in request.required_capabilities:
            agents = self.registry.find_agents_by_capability(capability)
            
            for agent_name in agents:
                if agent_name in request.exclude_agents:
                    continue
                
                registration = self.registry.get_registration(agent_name)
                if not registration:
                    continue
                
                # Calculate capability score
                capability_score = 0.0
                for cap in registration.capabilities:
                    if cap.name == capability:
                        capability_score = cap.confidence
                        break
                
                if capability_score > best_score:
                    best_score = capability_score
                    best_agent = agent_name
        
        if not best_agent:
            return None
        
        return DelegationDecision(
            agent_name=best_agent,
            confidence=best_score,
            reasoning=f"Selected based on highest capability match ({best_score:.2f})",
            estimated_execution_time=60.0,  # Default estimate
        )
    
    async def _performance_based_decision(
        self,
        request: DelegationRequest,
    ) -> Optional[DelegationDecision]:
        """Make decision based on agent performance metrics."""
        candidates = []
        
        for capability in request.required_capabilities:
            agents = self.registry.find_agents_by_capability(capability)
            candidates.extend(agents)
        
        # Remove duplicates and excluded agents
        candidates = list(set(candidates) - request.exclude_agents)
        
        if not candidates:
            return None
        
        best_agent = None
        best_score = 0.0
        
        for agent_name in candidates:
            registration = self.registry.get_registration(agent_name)
            if not registration:
                continue
            
            # Calculate performance score
            success_weight = 0.4
            speed_weight = 0.3
            usage_weight = 0.3
            
            success_score = registration.success_rate
            # Normalize response time (lower is better)
            speed_score = max(0.0, 1.0 - (registration.average_response_time / 60.0))
            # Normalize usage count (higher usage indicates reliability)
            usage_score = min(registration.usage_count / 100.0, 1.0)
            
            performance_score = (
                success_weight * success_score +
                speed_weight * speed_score +
                usage_weight * usage_score
            )
            
            if performance_score > best_score:
                best_score = performance_score
                best_agent = agent_name
        
        if not best_agent:
            return None
        
        return DelegationDecision(
            agent_name=best_agent,
            confidence=best_score,
            reasoning=f"Selected based on performance metrics (score: {best_score:.2f})",
            estimated_execution_time=self._estimate_execution_time(best_agent),
        )
    
    async def _load_balanced_decision(
        self,
        request: DelegationRequest,
    ) -> Optional[DelegationDecision]:
        """Make decision based on load balancing."""
        if not request.required_capabilities:
            return None
        
        # Use the first capability for load balancing
        primary_capability = request.required_capabilities[0]
        
        selected_agent = await self.manager.get_agent_for_capability(
            primary_capability,
            strategy="least_used"
        )
        
        if not selected_agent:
            return None
        
        return DelegationDecision(
            agent_name=selected_agent,
            confidence=0.8,  # Default confidence for load-balanced selection
            reasoning="Selected based on load balancing (least used agent)",
            estimated_execution_time=self._estimate_execution_time(selected_agent),
        )
    
    async def _hierarchical_decision(
        self,
        request: DelegationRequest,
    ) -> Optional[DelegationDecision]:
        """Make decision based on hierarchical delegation patterns."""
        # For hierarchical delegation, we consider agent relationships and delegation depth
        
        # Analyze request complexity
        complexity, estimated_time = self.request_analyzer.estimate_complexity(
            request.original_request,
            request.required_capabilities
        )
        
        # Select agent based on complexity and hierarchy
        if complexity == 'simple':
            # Use any capable agent
            return await self._capability_based_decision(request)
        elif complexity == 'moderate':
            # Prefer agents with good performance
            return await self._performance_based_decision(request)
        else:
            # For complex requests, use specialized agents or orchestrators
            specialized_agents = []
            for capability in request.required_capabilities:
                agents = self.registry.find_agents_by_capability(capability, min_confidence=0.8)
                specialized_agents.extend(agents)
            
            if specialized_agents:
                # Select the most specialized agent
                best_agent = specialized_agents[0]  # Simplified selection
                return DelegationDecision(
                    agent_name=best_agent,
                    confidence=0.9,
                    reasoning=f"Selected specialized agent for complex {complexity} request",
                    estimated_execution_time=estimated_time,
                )
        
        return None
    
    async def _hybrid_decision(
        self,
        request: DelegationRequest,
    ) -> Optional[DelegationDecision]:
        """Make decision using hybrid approach combining multiple strategies."""
        # Get decisions from multiple strategies
        strategies = [
            DelegationStrategy.CAPABILITY_BASED,
            DelegationStrategy.PERFORMANCE_BASED,
            DelegationStrategy.LOAD_BALANCED,
        ]
        
        decisions = []
        for strategy in strategies:
            try:
                decision = await self._make_delegation_decision(request, strategy)
                if decision:
                    decisions.append((strategy, decision))
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {e}")
        
        if not decisions:
            return None
        
        # Score and rank decisions
        scored_decisions = []
        for strategy, decision in decisions:
            # Weight different strategies
            strategy_weights = {
                DelegationStrategy.CAPABILITY_BASED: 0.4,
                DelegationStrategy.PERFORMANCE_BASED: 0.4,
                DelegationStrategy.LOAD_BALANCED: 0.2,
            }
            
            weight = strategy_weights.get(strategy, 0.3)
            score = decision.confidence * weight
            scored_decisions.append((score, decision, strategy))
        
        # Select the highest scoring decision
        scored_decisions.sort(reverse=True)
        best_score, best_decision, best_strategy = scored_decisions[0]
        
        # Update reasoning to reflect hybrid approach
        best_decision.reasoning = f"Hybrid selection (best: {best_strategy.value}, score: {best_score:.2f})"
        best_decision.confidence = best_score
        
        return best_decision
    
    async def _execute_delegation(
        self,
        request: DelegationRequest,
        decision: DelegationDecision,
    ) -> DelegationResult:
        """Execute the delegation to the selected agent.
        
        Args:
            request: Original delegation request
            decision: Delegation decision
            
        Returns:
            Delegation result
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            agent = self.registry.get_agent(decision.agent_name)
            if not agent:
                raise ValueError(f"Agent '{decision.agent_name}' not found")
            
            # Create execution context
            context = {
                'request_id': request.request_id,
                'delegation_depth': request.context.get('delegation_depth', 0) + 1,
                'original_context': request.context,
                'selected_by': 'delegation_engine',
            }
            
            # Execute the agent (mock execution for now)
            await asyncio.sleep(0.1)  # Simulate processing
            result = f"Agent {decision.agent_name} processed: {request.original_request[:50]}..."
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Record usage statistics
            self.registry.record_agent_usage(
                decision.agent_name,
                execution_time,
                success=True
            )
            
            return DelegationResult(
                request_id=request.request_id,
                success=True,
                selected_agent=decision.agent_name,
                delegation_path=[decision.agent_name],
                result=result,
                execution_time=execution_time,
                metadata={
                    'decision': decision.model_dump(),
                    'context': context,
                },
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Record failed usage
            self.registry.record_agent_usage(
                decision.agent_name,
                execution_time,
                success=False
            )
            
            raise e
    
    def _estimate_execution_time(self, agent_name: str) -> float:
        """Estimate execution time for an agent."""
        registration = self.registry.get_registration(agent_name)
        if registration and registration.average_response_time > 0:
            return registration.average_response_time
        return 60.0  # Default estimate
    
    def _update_performance_cache(
        self,
        agent_name: str,
        result: DelegationResult,
    ) -> None:
        """Update performance cache with delegation result."""
        if agent_name not in self._performance_cache:
            self._performance_cache[agent_name] = {}
        
        cache = self._performance_cache[agent_name]
        cache['last_execution_time'] = result.execution_time
        cache['last_success'] = result.success
        cache['last_updated'] = datetime.now().isoformat()
    
    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get comprehensive delegation statistics."""
        total_delegations = len(self._delegation_history)
        successful_delegations = sum(1 for r in self._delegation_history if r.success)
        
        avg_execution_time = 0.0
        if self._delegation_history:
            avg_execution_time = sum(r.execution_time for r in self._delegation_history) / total_delegations
        
        agent_usage = {}
        for result in self._delegation_history:
            if result.selected_agent:
                agent_usage[result.selected_agent] = agent_usage.get(result.selected_agent, 0) + 1
        
        return {
            'total_delegations': total_delegations,
            'successful_delegations': successful_delegations,
            'success_rate': successful_delegations / total_delegations if total_delegations > 0 else 0.0,
            'average_execution_time': avg_execution_time,
            'active_delegations': len(self._active_delegations),
            'agent_usage_distribution': agent_usage,
            'performance_cache_size': len(self._performance_cache),
        }

