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

"""Master orchestrator implementation using LlmAgent as the foundation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from google.genai import types
from pydantic import BaseModel, Field

from .base_orchestrator import BaseOrchestrator, OrchestratorConfig

# Import ADK components
try:
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.agents.sequential_agent import SequentialAgent
    from google.adk.agents.parallel_agent import ParallelAgent
    from google.adk.agents.loop_agent import LoopAgent
    from google.adk.tools.agent_tool import AgentTool
    from google.adk.tools.base_tool import BaseTool
except ImportError:
    # Fallback for development/testing
    LlmAgent = object
    SequentialAgent = object
    ParallelAgent = object
    LoopAgent = object
    AgentTool = object
    BaseTool = object

logger = logging.getLogger(__name__)


class RequestAnalysis(BaseModel):
    """Analysis of an incoming request."""
    
    intent: str = Field(description="Primary intent of the request")
    complexity: str = Field(description="Complexity level: simple, moderate, complex")
    required_capabilities: List[str] = Field(default_factory=list, description="Required agent capabilities")
    suggested_agents: List[str] = Field(default_factory=list, description="Suggested agents to handle request")
    workflow_type: str = Field(default="sequential", description="Suggested workflow type")
    estimated_duration: int = Field(default=30, description="Estimated duration in seconds")


class OrchestratorResponse(BaseModel):
    """Response from the orchestrator."""
    
    success: bool = Field(description="Whether the request was successful")
    result: Any = Field(description="The actual result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time: float = Field(description="Execution time in seconds")
    agents_used: List[str] = Field(default_factory=list, description="Agents that were used")
    workflow_type: str = Field(description="Type of workflow executed")


class MasterOrchestrator(BaseOrchestrator):
    """Master orchestrator that provides comprehensive agent orchestration.
    
    This orchestrator serves as the central hub for managing complex multi-agent
    workflows, providing intelligent delegation, state management, and execution
    coordination across different agent types and patterns.
    """
    
    def __init__(
        self,
        name: str = "master_orchestrator",
        model: str = "gemini-2.0-flash",
        description: Optional[str] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """Initialize the master orchestrator.
        
        Args:
            name: Name of the orchestrator
            model: LLM model to use for the orchestrator agent
            description: Description of orchestrator capabilities
            config: Configuration settings
        """
        super().__init__(
            name=name,
            description=description or "Master orchestrator for complex multi-agent workflows",
            config=config,
        )
        
        self.model = model
        self._llm_agent = None
        self._delegation_engine = None
        self._workflow_engine = None
        
        # Initialize core components
        self._initialize_components()
        
        logger.info(f"Initialized MasterOrchestrator with model: {model}")
    
    def _initialize_components(self) -> None:
        """Initialize core orchestrator components."""
        try:
            # Create the core LLM agent that powers the orchestrator
            self._llm_agent = self._create_orchestrator_agent()
            
            # Initialize delegation and workflow engines (will be implemented in later steps)
            # self._delegation_engine = DelegationEngine(self)
            # self._workflow_engine = WorkflowEngine(self)
            
            logger.info("Initialized orchestrator components")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator components: {e}")
            raise
    
    def _create_orchestrator_agent(self):
        """Create the core LLM agent for orchestration."""
        # This would create a proper LlmAgent in the real implementation
        orchestrator_instruction = """
        You are a master orchestrator responsible for managing complex multi-agent workflows.
        Your role is to:
        
        1. Analyze incoming requests to understand intent and complexity
        2. Determine the best agents and workflow patterns to handle each request
        3. Coordinate execution across multiple agents
        4. Manage state and context throughout the workflow
        5. Provide comprehensive responses with proper error handling
        
        You have access to various specialist agents and can orchestrate them using:
        - Sequential workflows for step-by-step processes
        - Parallel workflows for concurrent operations  
        - Loop workflows for iterative tasks
        - Direct agent delegation for simple requests
        
        Always consider the most efficient approach while ensuring quality results.
        """
        
        # Mock LLM agent for development
        class MockLlmAgent:
            def __init__(self, instruction: str, model: str):
                self.instruction = instruction
                self.model = model
                self.name = "orchestrator_llm"
                
            async def process(self, request: str, context: Dict[str, Any]) -> str:
                # Mock processing - in real implementation this would use the LLM
                return f"Processed request: {request[:100]}..."
        
        return MockLlmAgent(orchestrator_instruction, self.model)
    
    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResponse:
        """Process an incoming request through the orchestration system.
        
        Args:
            request: The user request to process
            context: Optional context information
            
        Returns:
            Orchestrator response with results and metadata
        """
        start_time = time.time()
        context = context or {}
        
        try:
            logger.info(f"Processing request: {request[:100]}...")
            self.metrics.total_requests += 1
            
            # Step 1: Analyze the request
            analysis = await self._analyze_request(request, context)
            logger.debug(f"Request analysis: {analysis}")
            
            # Step 2: Determine execution strategy
            execution_strategy = await self._determine_execution_strategy(analysis)
            logger.debug(f"Execution strategy: {execution_strategy}")
            
            # Step 3: Execute the workflow
            result = await self._execute_workflow(request, analysis, execution_strategy, context)
            
            # Step 4: Prepare response
            execution_time = time.time() - start_time
            response = OrchestratorResponse(
                success=True,
                result=result,
                metadata={
                    'analysis': analysis.model_dump(),
                    'execution_strategy': execution_strategy,
                    'context': context,
                },
                execution_time=execution_time,
                agents_used=analysis.suggested_agents,
                workflow_type=analysis.workflow_type,
            )
            
            self.metrics.successful_delegations += 1
            self._update_average_response_time(execution_time)
            
            logger.info(f"Successfully processed request in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.failed_delegations += 1
            
            logger.error(f"Failed to process request: {e}")
            
            return OrchestratorResponse(
                success=False,
                result=f"Error processing request: {str(e)}",
                metadata={'error': str(e), 'context': context},
                execution_time=execution_time,
                agents_used=[],
                workflow_type="error",
            )
    
    async def _analyze_request(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> RequestAnalysis:
        """Analyze the incoming request to determine processing strategy.
        
        Args:
            request: The user request
            context: Request context
            
        Returns:
            Analysis of the request
        """
        # In a real implementation, this would use the LLM agent to analyze the request
        # For now, provide a simple heuristic-based analysis
        
        request_lower = request.lower()
        
        # Determine complexity
        complexity = "simple"
        if any(word in request_lower for word in ["complex", "multiple", "analyze", "research"]):
            complexity = "complex"
        elif any(word in request_lower for word in ["compare", "process", "generate"]):
            complexity = "moderate"
        
        # Determine intent and capabilities
        intent = "general"
        required_capabilities = []
        suggested_agents = []
        
        if any(word in request_lower for word in ["search", "find", "research"]):
            intent = "research"
            required_capabilities.append("search")
            suggested_agents.extend(["research_agent", "search_agent"])
        
        if any(word in request_lower for word in ["analyze", "data", "process"]):
            intent = "analysis"
            required_capabilities.append("analysis")
            suggested_agents.extend(["analysis_agent", "data_agent"])
        
        if any(word in request_lower for word in ["write", "create", "generate"]):
            intent = "generation"
            required_capabilities.append("generation")
            suggested_agents.extend(["writer_agent", "content_agent"])
        
        # Determine workflow type
        workflow_type = "sequential"
        if "parallel" in request_lower or "simultaneously" in request_lower:
            workflow_type = "parallel"
        elif "loop" in request_lower or "iterate" in request_lower or "repeat" in request_lower:
            workflow_type = "loop"
        
        # Filter suggested agents to only include registered ones
        available_agents = [agent for agent in suggested_agents if agent in self._agents]
        if not available_agents:
            available_agents = list(self._agents.keys())[:2]  # Use first 2 available agents
        
        return RequestAnalysis(
            intent=intent,
            complexity=complexity,
            required_capabilities=required_capabilities,
            suggested_agents=available_agents,
            workflow_type=workflow_type,
            estimated_duration=30 if complexity == "simple" else 60 if complexity == "moderate" else 120,
        )
    
    async def _determine_execution_strategy(
        self,
        analysis: RequestAnalysis,
    ) -> Dict[str, Any]:
        """Determine the execution strategy based on request analysis.
        
        Args:
            analysis: Request analysis
            
        Returns:
            Execution strategy configuration
        """
        strategy = {
            'type': analysis.workflow_type,
            'agents': analysis.suggested_agents,
            'timeout': analysis.estimated_duration,
            'retry_attempts': self.config.retry_attempts,
        }
        
        # Add workflow-specific configuration
        if analysis.workflow_type == "parallel":
            strategy['max_concurrent'] = min(len(analysis.suggested_agents), 3)
        elif analysis.workflow_type == "loop":
            strategy['max_iterations'] = 5
            strategy['termination_condition'] = "success_or_max_iterations"
        
        return strategy
    
    async def _execute_workflow(
        self,
        request: str,
        analysis: RequestAnalysis,
        strategy: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """Execute the workflow based on the determined strategy.
        
        Args:
            request: Original request
            analysis: Request analysis
            strategy: Execution strategy
            context: Request context
            
        Returns:
            Workflow execution result
        """
        workflow_type = strategy['type']
        agents = strategy['agents']
        
        if not agents:
            return await self._direct_llm_processing(request, context)
        
        if workflow_type == "sequential":
            return await self._execute_sequential_workflow(request, agents, context)
        elif workflow_type == "parallel":
            return await self._execute_parallel_workflow(request, agents, context)
        elif workflow_type == "loop":
            return await self._execute_loop_workflow(request, agents, strategy, context)
        else:
            # Default to direct agent delegation
            return await self._execute_direct_delegation(request, agents[0], context)
    
    async def _direct_llm_processing(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> str:
        """Process request directly with the orchestrator's LLM agent.
        
        Args:
            request: The request to process
            context: Request context
            
        Returns:
            LLM response
        """
        if self._llm_agent:
            return await self._llm_agent.process(request, context)
        else:
            return f"Direct processing of: {request}"
    
    async def _execute_sequential_workflow(
        self,
        request: str,
        agents: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a sequential workflow with the specified agents.
        
        Args:
            request: The request to process
            agents: List of agent names to use
            context: Request context
            
        Returns:
            Sequential workflow results
        """
        results = []
        current_input = request
        
        for agent_name in agents:
            agent = self.get_agent(agent_name)
            if agent:
                try:
                    # Mock agent execution
                    result = f"Agent {agent_name} processed: {current_input[:50]}..."
                    results.append({
                        'agent': agent_name,
                        'input': current_input,
                        'output': result,
                        'success': True,
                    })
                    current_input = result  # Chain the output to next agent
                except Exception as e:
                    results.append({
                        'agent': agent_name,
                        'input': current_input,
                        'output': f"Error: {str(e)}",
                        'success': False,
                    })
                    break
        
        return {
            'workflow_type': 'sequential',
            'results': results,
            'final_output': results[-1]['output'] if results else "No results",
        }
    
    async def _execute_parallel_workflow(
        self,
        request: str,
        agents: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a parallel workflow with the specified agents.
        
        Args:
            request: The request to process
            agents: List of agent names to use
            context: Request context
            
        Returns:
            Parallel workflow results
        """
        tasks = []
        
        for agent_name in agents:
            agent = self.get_agent(agent_name)
            if agent:
                # Create async task for each agent
                task = self._execute_agent_async(agent_name, request, context)
                tasks.append((agent_name, task))
        
        # Execute all tasks concurrently
        results = []
        if tasks:
            task_results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for (agent_name, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    results.append({
                        'agent': agent_name,
                        'output': f"Error: {str(result)}",
                        'success': False,
                    })
                else:
                    results.append({
                        'agent': agent_name,
                        'output': result,
                        'success': True,
                    })
        
        return {
            'workflow_type': 'parallel',
            'results': results,
            'combined_output': [r['output'] for r in results if r['success']],
        }
    
    async def _execute_loop_workflow(
        self,
        request: str,
        agents: List[str],
        strategy: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a loop workflow with the specified agents.
        
        Args:
            request: The request to process
            agents: List of agent names to use
            strategy: Execution strategy with loop configuration
            context: Request context
            
        Returns:
            Loop workflow results
        """
        max_iterations = strategy.get('max_iterations', 5)
        results = []
        current_input = request
        
        for iteration in range(max_iterations):
            iteration_results = []
            
            for agent_name in agents:
                agent = self.get_agent(agent_name)
                if agent:
                    try:
                        result = f"Iteration {iteration + 1} - Agent {agent_name}: {current_input[:30]}..."
                        iteration_results.append({
                            'agent': agent_name,
                            'output': result,
                            'success': True,
                        })
                        current_input = result
                    except Exception as e:
                        iteration_results.append({
                            'agent': agent_name,
                            'output': f"Error: {str(e)}",
                            'success': False,
                        })
            
            results.append({
                'iteration': iteration + 1,
                'results': iteration_results,
            })
            
            # Check termination condition
            if all(r['success'] for r in iteration_results):
                # Simple success condition - could be more sophisticated
                if "complete" in current_input.lower() or iteration >= 2:
                    break
        
        return {
            'workflow_type': 'loop',
            'iterations': len(results),
            'results': results,
            'final_output': current_input,
        }
    
    async def _execute_direct_delegation(
        self,
        request: str,
        agent_name: str,
        context: Dict[str, Any],
    ) -> str:
        """Execute direct delegation to a single agent.
        
        Args:
            request: The request to process
            agent_name: Name of the agent to delegate to
            context: Request context
            
        Returns:
            Agent response
        """
        return await self._execute_agent_async(agent_name, request, context)
    
    async def _execute_agent_async(
        self,
        agent_name: str,
        request: str,
        context: Dict[str, Any],
    ) -> str:
        """Execute an agent asynchronously.
        
        Args:
            agent_name: Name of the agent to execute
            request: The request to process
            context: Request context
            
        Returns:
            Agent response
        """
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        # Mock agent execution - in real implementation this would call the actual agent
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Agent {agent_name} response to: {request[:50]}..."
    
    def _update_average_response_time(self, execution_time: float) -> None:
        """Update the average response time metric.
        
        Args:
            execution_time: Latest execution time
        """
        total_requests = self.metrics.total_requests
        if total_requests == 1:
            self.metrics.average_response_time = execution_time
        else:
            # Calculate running average
            current_avg = self.metrics.average_response_time
            self.metrics.average_response_time = (
                (current_avg * (total_requests - 1) + execution_time) / total_requests
            )
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the orchestrator.
        
        Returns:
            Status information including metrics and configuration
        """
        return {
            'name': self.name,
            'description': self.description,
            'model': self.model,
            'config': self.config.model_dump(),
            'metrics': self.metrics.model_dump(),
            'registered_agents': self.list_agents(),
            'active_contexts': len(self._active_contexts),
            'state_entries': len(self._state_store),
        }

