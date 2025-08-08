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

"""Main orchestrator agent for managing and coordinating multiple agents."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from google.genai import types
from typing_extensions import override

from ..agents.base_agent import BaseAgent
from ..agents.base_agent_config import BaseAgentConfig
from ..agents.callback_context import CallbackContext
from ..agents.invocation_context import InvocationContext
from ..agents.llm_agent import LlmAgent
from ..agents.loop_agent import LoopAgent
from ..agents.parallel_agent import ParallelAgent
from ..agents.sequential_agent import SequentialAgent
from ..events.event import Event
from ..tools.agent_tool import AgentTool
from .environment_manager import EnvironmentManager
from .execution_tracker import ExecutionTracker
from .models.agent_metadata import AgentMetadata
from .models.agent_metadata import AgentStatus
from .models.agent_template import AgentTemplate
from .models.execution_context import ExecutionContext
from .models.execution_context import ExecutionMode
from .registry.agent_registry import AgentRegistry
from .template_manager import TemplateManager

logger = logging.getLogger('google_adk.orchestrator')


class OrchestratorAgentConfig(BaseAgentConfig):
  """Configuration for the orchestrator agent."""
  
  # Core orchestrator settings
  max_concurrent_agents: int = 10
  """Maximum number of agents that can run concurrently."""
  
  default_timeout: float = 300.0
  """Default timeout for agent execution in seconds."""
  
  enable_tracing: bool = True
  """Whether to enable execution tracing."""
  
  enable_debugging: bool = False
  """Whether to enable debug mode."""
  
  # Environment settings
  default_environment: str = "local"
  """Default execution environment."""
  
  supported_environments: List[str] = ["local", "wsl2", "ssh"]
  """List of supported environments."""
  
  # Template and registry settings
  template_storage_path: Optional[str] = None
  """Path for template storage (None for in-memory)."""
  
  auto_discover_agents: bool = True
  """Whether to automatically discover available agents."""


class OrchestratorAgent(LlmAgent):
  """Advanced orchestrator agent for managing complex multi-agent workflows.
  
  The OrchestratorAgent serves as a high-level coordinator that can:
  - Create, modify, and delete agent templates
  - Instantiate agents from templates dynamically
  - Execute agents asynchronously with full tracing
  - Manage different execution environments (local, WSL2, SSH)
  - Coordinate sequential, parallel, and loop workflows
  - Provide debugging and analysis capabilities
  - Integrate with MCP tools for enhanced functionality
  
  This agent leverages the existing ADK patterns including:
  - Hierarchical delegation with LlmAgent as the foundation
  - Agent-as-Tool architecture using AgentTool
  - Callback mechanisms for flow control
  - A2A communication with AgentCardBuilder
  """
  
  config_type: Type[BaseAgentConfig] = OrchestratorAgentConfig
  
  def __init__(
      self,
      *,
      name: str = "orchestrator",
      description: str = "Advanced agent orchestrator for managing complex workflows",
      model: str = "gemini-2.0-flash",
      instruction: Optional[str] = None,
      template_manager: Optional[TemplateManager] = None,
      environment_manager: Optional[EnvironmentManager] = None,
      execution_tracker: Optional[ExecutionTracker] = None,
      agent_registry: Optional[AgentRegistry] = None,
      **kwargs
  ):
    # Set default instruction if not provided
    if instruction is None:
      instruction = self._get_default_instruction()
    
    # Initialize core managers
    self.template_manager = template_manager or TemplateManager()
    self.environment_manager = environment_manager or EnvironmentManager()
    self.execution_tracker = execution_tracker or ExecutionTracker()
    self.agent_registry = agent_registry or AgentRegistry()
    
    # Initialize orchestrator tools
    orchestrator_tools = self._create_orchestrator_tools()
    
    # Merge with any provided tools
    tools = kwargs.get('tools', [])
    if isinstance(tools, list):
      tools.extend(orchestrator_tools)
    else:
      tools = orchestrator_tools
    kwargs['tools'] = tools
    
    # Initialize the LlmAgent
    super().__init__(
        name=name,
        description=description,
        model=model,
        instruction=instruction,
        **kwargs
    )
    
    # Set up callbacks for orchestration
    self.before_agent_callback = self._before_agent_execution
    self.after_agent_callback = self._after_agent_execution
    
    logger.info(f"Orchestrator agent '{name}' initialized")
  
  def _get_default_instruction(self) -> str:
    """Get the default instruction for the orchestrator."""
    return """You are an advanced AI agent orchestrator with the following capabilities:

CORE FUNCTIONS:
1. Template Management: Create, modify, delete, and instantiate agent templates
2. Environment Management: Handle local, WSL2, and SSH execution environments  
3. Workflow Orchestration: Coordinate sequential, parallel, and loop agent workflows
4. Async Execution: Run multiple agents concurrently with full tracing
5. Debugging & Analysis: Provide comprehensive debugging and analysis tools
6. MCP Integration: Leverage Model Context Protocol tools for enhanced capabilities

DELEGATION STRATEGY:
- Analyze user requests to determine the best approach (sequential, parallel, or loop)
- Create specialized agents for specific tasks when needed
- Use existing templates when appropriate
- Coordinate complex workflows across multiple environments

AVAILABLE TOOLS:
- create_agent_template: Create new agent templates
- instantiate_agent: Create agent instances from templates
- execute_agent_async: Run agents asynchronously with monitoring
- list_templates: Show available agent templates
- get_execution_status: Check status of running agents
- set_environment: Switch execution environments
- debug_agent: Debug and analyze agent behavior

Always explain your reasoning and provide clear status updates on orchestration activities."""
  
  def _create_orchestrator_tools(self) -> List[Any]:
    """Create the core orchestrator tools."""
    tools = []
    
    # Template management tools
    def create_agent_template(
        name: str,
        agent_type: str,
        description: str = "",
        agent_config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
      """Create a new agent template.
      
      Args:
        name: Name of the template
        agent_type: Type of agent (LlmAgent, SequentialAgent, ParallelAgent, LoopAgent)
        description: Description of what the agent does
        agent_config: Agent configuration dictionary
        tags: Tags for categorizing the template
      
      Returns:
        Template ID of the created template
      """
      template = AgentTemplate(
          name=name,
          agent_type=agent_type,
          description=description,
          agent_config=agent_config or {},
          tags=tags or []
      )
      return self.template_manager.create_template(template)
    
    def list_templates(tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
      """List available agent templates.
      
      Args:
        tags: Optional tags to filter by
      
      Returns:
        List of template summaries
      """
      templates = self.template_manager.list_templates(tags=tags)
      return [
          {
              "id": t.id,
              "name": t.name,
              "version": t.version,
              "agent_type": t.agent_type,
              "description": t.description,
              "tags": t.tags,
              "usage_count": t.usage_count
          }
          for t in templates
      ]
    
    def instantiate_agent(
        template_id: str,
        agent_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
      """Instantiate an agent from a template.
      
      Args:
        template_id: ID of the template to use
        agent_name: Name for the agent instance
        config_overrides: Configuration overrides
      
      Returns:
        Agent ID of the created instance
      """
      return self.template_manager.instantiate_agent(
          template_id=template_id,
          agent_name=agent_name,
          config_overrides=config_overrides or {}
      )
    
    def execute_agent_async(
        agent_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        environment: Optional[str] = None
    ) -> str:
      """Execute an agent asynchronously.
      
      Args:
        agent_id: ID of the agent to execute
        input_data: Input data for the agent
        timeout: Execution timeout in seconds
        environment: Execution environment (local, wsl2, ssh)
      
      Returns:
        Execution ID for tracking
      """
      return asyncio.create_task(
          self.execution_tracker.execute_agent_async(
              agent_id=agent_id,
              input_data=input_data or {},
              timeout=timeout,
              environment=environment or "local"
          )
      ).get_name() or "unknown"
    
    def get_execution_status(execution_id: str) -> Dict[str, Any]:
      """Get the status of an executing agent.
      
      Args:
        execution_id: ID of the execution to check
      
      Returns:
        Execution status information
      """
      return self.execution_tracker.get_execution_status(execution_id)
    
    def set_environment(environment: str) -> str:
      """Set the execution environment.
      
      Args:
        environment: Environment to use (local, wsl2, ssh)
      
      Returns:
        Confirmation message
      """
      if self.environment_manager.set_active_environment(environment):
        return f"Environment set to {environment}"
      else:
        return f"Failed to set environment to {environment}"
    
    def debug_agent(agent_id: str) -> Dict[str, Any]:
      """Debug and analyze an agent.
      
      Args:
        agent_id: ID of the agent to debug
      
      Returns:
        Debug information
      """
      metadata = self.agent_registry.get_agent_metadata(agent_id)
      if metadata:
        return {
            "agent_id": agent_id,
            "status": metadata.status.value,
            "capabilities": [cap.name for cap in metadata.capabilities],
            "runtime_summary": metadata.get_runtime_summary(),
            "debug_enabled": metadata.debug_enabled
        }
      return {"error": f"Agent {agent_id} not found"}
    
    # Add tools to the list
    tools.extend([
        create_agent_template,
        list_templates,
        instantiate_agent,
        execute_agent_async,
        get_execution_status,
        set_environment,
        debug_agent
    ])
    
    return tools
  
  async def _before_agent_execution(self, callback_context: CallbackContext) -> Optional[types.Content]:
    """Callback executed before agent runs."""
    logger.debug(f"Orchestrator: Before execution callback for {callback_context.agent.name}")
    
    # Update execution tracking
    if hasattr(callback_context, 'execution_id'):
      self.execution_tracker.update_execution_status(
          callback_context.execution_id,
          "starting"
      )
    
    return None
  
  async def _after_agent_execution(self, callback_context: CallbackContext) -> Optional[types.Content]:
    """Callback executed after agent runs."""
    logger.debug(f"Orchestrator: After execution callback for {callback_context.agent.name}")
    
    # Update execution tracking
    if hasattr(callback_context, 'execution_id'):
      self.execution_tracker.update_execution_status(
          callback_context.execution_id,
          "completed"
      )
    
    return None
  
  def create_sequential_workflow(
      self,
      name: str,
      agent_templates: List[str],
      description: str = ""
  ) -> str:
    """Create a sequential workflow from agent templates.
    
    Args:
      name: Name of the workflow
      agent_templates: List of template IDs to execute in sequence
      description: Description of the workflow
    
    Returns:
      Template ID of the created workflow
    """
    # Instantiate sub-agents from templates
    sub_agents = []
    for template_id in agent_templates:
      agent_id = self.template_manager.instantiate_agent(template_id)
      agent = self.agent_registry.get_agent(agent_id)
      if agent:
        sub_agents.append(agent)
    
    # Create sequential agent configuration
    config = {
        "name": name,
        "description": description,
        "sub_agents": [agent.name for agent in sub_agents]
    }
    
    # Create template
    template = AgentTemplate(
        name=name,
        agent_type="SequentialAgent",
        description=description,
        agent_config=config,
        tags=["workflow", "sequential"]
    )
    
    return self.template_manager.create_template(template)
  
  def create_parallel_workflow(
      self,
      name: str,
      agent_templates: List[str],
      description: str = ""
  ) -> str:
    """Create a parallel workflow from agent templates.
    
    Args:
      name: Name of the workflow
      agent_templates: List of template IDs to execute in parallel
      description: Description of the workflow
    
    Returns:
      Template ID of the created workflow
    """
    # Instantiate sub-agents from templates
    sub_agents = []
    for template_id in agent_templates:
      agent_id = self.template_manager.instantiate_agent(template_id)
      agent = self.agent_registry.get_agent(agent_id)
      if agent:
        sub_agents.append(agent)
    
    # Create parallel agent configuration
    config = {
        "name": name,
        "description": description,
        "sub_agents": [agent.name for agent in sub_agents]
    }
    
    # Create template
    template = AgentTemplate(
        name=name,
        agent_type="ParallelAgent",
        description=description,
        agent_config=config,
        tags=["workflow", "parallel"]
    )
    
    return self.template_manager.create_template(template)
  
  def create_loop_workflow(
      self,
      name: str,
      agent_templates: List[str],
      max_iterations: int = 10,
      description: str = ""
  ) -> str:
    """Create a loop workflow from agent templates.
    
    Args:
      name: Name of the workflow
      agent_templates: List of template IDs to execute in a loop
      max_iterations: Maximum number of loop iterations
      description: Description of the workflow
    
    Returns:
      Template ID of the created workflow
    """
    # Instantiate sub-agents from templates
    sub_agents = []
    for template_id in agent_templates:
      agent_id = self.template_manager.instantiate_agent(template_id)
      agent = self.agent_registry.get_agent(agent_id)
      if agent:
        sub_agents.append(agent)
    
    # Create loop agent configuration
    config = {
        "name": name,
        "description": description,
        "sub_agents": [agent.name for agent in sub_agents],
        "max_iterations": max_iterations
    }
    
    # Create template
    template = AgentTemplate(
        name=name,
        agent_type="LoopAgent",
        description=description,
        agent_config=config,
        tags=["workflow", "loop"]
    )
    
    return self.template_manager.create_template(template)
  
  async def orchestrate_complex_workflow(
      self,
      workflow_definition: Dict[str, Any]
  ) -> str:
    """Orchestrate a complex workflow with multiple patterns.
    
    Args:
      workflow_definition: Complex workflow definition
    
    Returns:
      Execution ID for the orchestrated workflow
    """
    # This method would handle complex workflow orchestration
    # combining sequential, parallel, and loop patterns as needed
    logger.info(f"Orchestrating complex workflow: {workflow_definition.get('name', 'unnamed')}")
    
    # Implementation would parse the workflow definition and create
    # the appropriate agent hierarchy using the existing patterns
    
    return "complex_workflow_execution_id"
  
  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Implementation of async execution for the orchestrator."""
    logger.info(f"Orchestrator {self.name} starting execution")
    
    # Delegate to the parent LlmAgent implementation
    async for event in super()._run_async_impl(ctx):
      yield event
    
    logger.info(f"Orchestrator {self.name} completed execution")

