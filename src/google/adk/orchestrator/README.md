# ADK Orchestrator Agent

The ADK Orchestrator Agent is an advanced agent orchestration and management system that provides comprehensive capabilities for creating, managing, and coordinating complex multi-agent workflows.

## 🎯 Core Capabilities

### Template Management
- **Create, modify, and delete agent templates** with full CRUD operations
- **Template versioning and inheritance** for maintainable agent configurations
- **Template validation** with comprehensive error checking and best practices
- **Template discovery and search** with tag-based filtering

### Environment Abstraction
- **Multi-environment support**: Local, WSL2, and SSH environments
- **Environment detection and switching** with automatic capability detection
- **Consistent API** across all environments for commands and file operations
- **Environment-specific optimizations** and error handling

### Workflow Orchestration
- **Sequential workflows**: Execute agents in order with dependency management
- **Parallel workflows**: Run agents concurrently with result aggregation
- **Loop workflows**: Iterative agent execution with termination conditions
- **Complex workflows**: Combine multiple patterns for sophisticated orchestration

### Async Execution & Monitoring
- **Asynchronous agent execution** with full concurrency control
- **Real-time progress tracking** and status monitoring
- **Execution history and metrics** for performance analysis
- **Resource management** with configurable limits and timeouts

### Agent Registry & Discovery
- **Dynamic agent registration** with metadata tracking
- **Capability-based discovery** for finding suitable agents
- **Lifecycle management** with status tracking and cleanup
- **Performance metrics** and usage statistics

## 🏗️ Architecture

The orchestrator follows a modular architecture with clear separation of concerns:

```
OrchestratorAgent (extends LlmAgent)
├── TemplateManager      # Template CRUD and validation
├── EnvironmentManager   # Multi-environment abstraction
├── ExecutionTracker     # Async execution monitoring
├── AgentRegistry        # Agent instance management
└── Tools               # Built-in orchestrator tools
```

### Key Components

- **OrchestratorAgent**: Main coordinator extending LlmAgent with orchestration capabilities
- **TemplateManager**: Handles template storage, validation, and instantiation
- **EnvironmentManager**: Provides abstraction over execution environments
- **ExecutionTracker**: Monitors and manages asynchronous agent executions
- **AgentRegistry**: Maintains registry of agent instances and their metadata

## 🚀 Quick Start

### Basic Usage

```python
from google.adk.orchestrator import OrchestratorAgent
from google.adk.orchestrator.models.agent_template import AgentTemplate

# Create orchestrator
orchestrator = OrchestratorAgent(
    name="my_orchestrator",
    model="gemini-2.0-flash"
)

# Create an agent template
template = AgentTemplate(
    name="research_agent",
    agent_type="LlmAgent",
    description="Specialized research agent",
    agent_config={
        "name": "research_agent",
        "model": "gemini-2.0-flash",
        "instruction": "You are a research specialist.",
        "tools": ["google_search"]
    },
    tags=["research", "information"]
)

# Store the template
template_id = orchestrator.template_manager.create_template(template)

# Instantiate an agent from the template
agent_id = orchestrator.template_manager.instantiate_agent(template_id)

# Execute the agent asynchronously
execution_id = await orchestrator.execute_agent_async(
    agent_id=agent_id,
    input_data={"query": "Latest AI research trends"},
    environment="local"
)

# Monitor execution
status = orchestrator.execution_tracker.get_execution_status(execution_id)
print(f"Execution status: {status}")
```

### Creating Workflows

```python
# Create a sequential workflow
workflow_id = orchestrator.create_sequential_workflow(
    name="research_analysis_workflow",
    agent_templates=[research_template_id, analysis_template_id],
    description="Research then analyze findings"
)

# Create a parallel workflow
parallel_workflow_id = orchestrator.create_parallel_workflow(
    name="multi_source_research",
    agent_templates=[web_research_id, paper_research_id, news_research_id],
    description="Research from multiple sources simultaneously"
)

# Create a loop workflow
loop_workflow_id = orchestrator.create_loop_workflow(
    name="iterative_refinement",
    agent_templates=[draft_agent_id, review_agent_id],
    max_iterations=5,
    description="Iteratively improve content through drafting and review"
)
```

### Environment Management

```python
# List available environments
environments = orchestrator.environment_manager.get_available_environments()
print(f"Available: {environments}")

# Switch environment
orchestrator.environment_manager.set_active_environment("wsl2")

# Add SSH environment
orchestrator.environment_manager.add_ssh_environment(
    name="production_server",
    host="prod.example.com",
    username="deploy",
    key_file="/path/to/key.pem"
)

# Execute command in specific environment
result = await orchestrator.environment_manager.execute_command(
    command="python --version",
    environment="wsl2"
)
```

## 🛠️ Built-in Tools

The orchestrator provides several built-in tools accessible through natural language:

- `create_agent_template`: Create new agent templates
- `list_templates`: Show available templates with filtering
- `instantiate_agent`: Create agent instances from templates
- `execute_agent_async`: Run agents asynchronously
- `get_execution_status`: Check execution progress
- `set_environment`: Switch execution environments
- `debug_agent`: Debug and analyze agent behavior

## 📊 Monitoring & Analytics

### Execution Statistics
```python
stats = orchestrator.execution_tracker.get_execution_statistics()
# Returns: active_executions, success_rate, avg_execution_time, etc.
```

### Template Analytics
```python
template_stats = orchestrator.template_manager.get_template_statistics()
# Returns: total_templates, usage_counts, popular_templates, etc.
```

### Registry Insights
```python
registry_stats = orchestrator.agent_registry.get_registry_statistics()
# Returns: total_agents, status_distribution, capabilities, etc.
```

## 🔧 Configuration

### Orchestrator Configuration

```python
from google.adk.orchestrator.orchestrator_agent import OrchestratorAgentConfig

config = OrchestratorAgentConfig(
    max_concurrent_agents=20,
    default_timeout=600.0,
    enable_tracing=True,
    enable_debugging=True,
    default_environment="local",
    supported_environments=["local", "wsl2", "ssh"],
    template_storage_path="/path/to/templates",
    auto_discover_agents=True
)

orchestrator = OrchestratorAgent(
    name="configured_orchestrator",
    config=config
)
```

### Environment-Specific Settings

```python
# Configure WSL2 with specific distribution
wsl_env = WSLEnvironment(distribution="Ubuntu-20.04")

# Configure SSH with key authentication
ssh_env = SSHEnvironment(
    host="remote.example.com",
    username="user",
    key_file="/path/to/private_key",
    port=2222
)
```

## 🔍 Debugging & Tracing

### Enable Debug Mode
```python
orchestrator = OrchestratorAgent(
    name="debug_orchestrator",
    enable_debugging=True,
    enable_tracing=True
)
```

### Debug Agent Execution
```python
# Get detailed agent information
debug_info = orchestrator.debug_agent(agent_id)
print(f"Agent status: {debug_info['status']}")
print(f"Capabilities: {debug_info['capabilities']}")
print(f"Runtime summary: {debug_info['runtime_summary']}")
```

### Execution Tracing
```python
# Get execution history
history = orchestrator.execution_tracker.get_execution_history(limit=50)

# Get active executions
active = orchestrator.execution_tracker.get_active_executions()
```

## 🔒 Security Considerations

- **Environment Isolation**: Each environment provides isolated execution
- **Template Validation**: Comprehensive validation prevents malicious templates
- **Resource Limits**: Configurable limits prevent resource exhaustion
- **Access Control**: Environment-specific access controls and authentication

## 🧪 Testing

Run the example to test orchestrator functionality:

```bash
python examples/orchestrator_example.py
```

## 📚 API Reference

### OrchestratorAgent

Main orchestrator class extending LlmAgent with orchestration capabilities.

**Methods:**
- `create_sequential_workflow(name, agent_templates, description)`: Create sequential workflow
- `create_parallel_workflow(name, agent_templates, description)`: Create parallel workflow  
- `create_loop_workflow(name, agent_templates, max_iterations, description)`: Create loop workflow
- `orchestrate_complex_workflow(workflow_definition)`: Handle complex workflow patterns

### TemplateManager

Manages agent templates with CRUD operations and validation.

**Methods:**
- `create_template(template)`: Create new template
- `get_template(template_id)`: Get template by ID
- `list_templates(tags, agent_type, author)`: List templates with filtering
- `instantiate_agent(template_id, agent_name, config_overrides)`: Create agent from template
- `clone_template(template_id, new_name, modifications)`: Clone existing template

### EnvironmentManager

Provides abstraction over different execution environments.

**Methods:**
- `get_available_environments()`: List available environments
- `set_active_environment(environment)`: Switch active environment
- `execute_command(command, environment, timeout, working_dir)`: Execute command
- `add_ssh_environment(name, host, username, ...)`: Add SSH environment

### ExecutionTracker

Monitors and manages asynchronous agent executions.

**Methods:**
- `execute_agent_async(agent_id, input_data, timeout, environment)`: Execute agent
- `get_execution_status(execution_id)`: Get execution status
- `cancel_execution(execution_id, reason)`: Cancel execution
- `get_execution_statistics()`: Get execution metrics

### AgentRegistry

Manages agent instances and their metadata.

**Methods:**
- `register_agent(agent, agent_id, template_id, environment)`: Register agent
- `get_agent(agent_id)`: Get agent instance
- `list_agents(status, environment, template_id)`: List agents with filtering
- `find_agents_by_capability(capability)`: Find agents by capability

## 🤝 Contributing

The orchestrator is designed to be extensible. Key extension points:

- **Custom Environments**: Implement `BaseEnvironment` for new execution contexts
- **Storage Backends**: Extend `TemplateStorage` for different persistence layers
- **Validation Rules**: Add custom validation logic to `TemplateValidator`
- **Monitoring**: Extend `ExecutionTracker` with custom metrics and alerts

## 📄 License

Licensed under the Apache License, Version 2.0. See the LICENSE file for details.

