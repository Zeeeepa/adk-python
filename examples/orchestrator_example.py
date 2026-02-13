#!/usr/bin/env python3
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

"""Example demonstrating the ADK Orchestrator Agent capabilities."""

import asyncio
import logging
from pathlib import Path
import sys

# Add the src directory to the path so we can import the orchestrator
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from google.adk.orchestrator import OrchestratorAgent
from google.adk.orchestrator.models.agent_template import AgentTemplate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating orchestrator capabilities."""
    
    print("🚀 ADK Orchestrator Agent Example")
    print("=" * 50)
    
    # Create the orchestrator agent
    orchestrator = OrchestratorAgent(
        name="example_orchestrator",
        description="Example orchestrator for demonstration",
        model="gemini-2.0-flash"
    )
    
    print(f"✅ Created orchestrator: {orchestrator.name}")
    
    # Example 1: Create a simple agent template
    print("\n📝 Creating agent templates...")
    
    # Create a research agent template
    research_template = AgentTemplate(
        name="research_agent",
        agent_type="LlmAgent",
        description="Agent specialized in research and information gathering",
        agent_config={
            "name": "research_agent",
            "model": "gemini-2.0-flash",
            "instruction": "You are a research specialist. Help users find and analyze information.",
            "tools": ["google_search", "web_scraper"]
        },
        tags=["research", "information", "analysis"]
    )
    
    research_template_id = orchestrator.template_manager.create_template(research_template)
    print(f"✅ Created research agent template: {research_template_id}")
    
    # Create an analysis agent template
    analysis_template = AgentTemplate(
        name="analysis_agent", 
        agent_type="LlmAgent",
        description="Agent specialized in data analysis and insights",
        agent_config={
            "name": "analysis_agent",
            "model": "gemini-2.0-flash",
            "instruction": "You are a data analysis expert. Help users understand and interpret data.",
            "tools": ["data_processor", "chart_generator"]
        },
        tags=["analysis", "data", "insights"]
    )
    
    analysis_template_id = orchestrator.template_manager.create_template(analysis_template)
    print(f"✅ Created analysis agent template: {analysis_template_id}")
    
    # Example 2: List available templates
    print("\n📋 Listing available templates...")
    templates = orchestrator.template_manager.list_templates()
    for template in templates:
        print(f"  - {template.name} v{template.version} ({template.agent_type})")
        print(f"    Description: {template.description}")
        print(f"    Tags: {', '.join(template.tags)}")
        print()
    
    # Example 3: Create a sequential workflow
    print("🔄 Creating sequential workflow...")
    workflow_template_id = orchestrator.create_sequential_workflow(
        name="research_analysis_workflow",
        agent_templates=[research_template_id, analysis_template_id],
        description="Sequential workflow that researches a topic then analyzes the findings"
    )
    print(f"✅ Created sequential workflow: {workflow_template_id}")
    
    # Example 4: Environment management
    print("\n🌍 Environment management...")
    available_envs = orchestrator.environment_manager.get_available_environments()
    print(f"Available environments: {', '.join(available_envs)}")
    
    active_env = orchestrator.environment_manager.get_active_environment()
    print(f"Active environment: {active_env}")
    
    # Get environment info
    env_info = orchestrator.environment_manager.get_environment_info()
    print(f"Environment info: {env_info}")
    
    # Example 5: Template statistics
    print("\n📊 Template statistics...")
    stats = orchestrator.template_manager.get_template_statistics()
    print(f"Total templates: {stats['total_templates']}")
    print(f"Templates by type: {stats['by_type']}")
    print(f"Total usage: {stats['total_usage']}")
    
    # Example 6: Execution tracking
    print("\n⚡ Execution tracking...")
    exec_stats = orchestrator.execution_tracker.get_execution_statistics()
    print(f"Active executions: {exec_stats['active_executions']}")
    print(f"Max concurrent limit: {exec_stats['max_concurrent_limit']}")
    print(f"Success rate: {exec_stats['success_rate']}%")
    
    # Example 7: Agent registry
    print("\n📚 Agent registry...")
    registry_stats = orchestrator.agent_registry.get_registry_statistics()
    print(f"Total registered agents: {registry_stats['total_agents']}")
    print(f"Status distribution: {registry_stats['status_distribution']}")
    
    print("\n🎉 Orchestrator example completed successfully!")
    print("\nThe orchestrator provides:")
    print("  ✅ Template management with CRUD operations")
    print("  ✅ Environment abstraction (local, WSL2, SSH)")
    print("  ✅ Workflow orchestration (sequential, parallel, loop)")
    print("  ✅ Async execution with monitoring")
    print("  ✅ Agent registry and discovery")
    print("  ✅ Comprehensive tracing and debugging")


if __name__ == "__main__":
    asyncio.run(main())

