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

"""Research Orchestrator Example.

This example demonstrates a comprehensive research orchestrator that uses
the ADK framework to coordinate multiple specialized agents for complex
research workflows.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import MasterOrchestrator, AgentRegistry, AgentManager
from orchestrator.delegation_engine import DelegationEngine, DelegationRequest, DelegationPriority
from orchestrator.request_analyzer import AdvancedRequestAnalyzer

# Mock ADK agents for demonstration
class MockAgent:
    """Mock agent for demonstration purposes."""
    
    def __init__(self, name: str, description: str, capabilities: list = None):
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
    
    async def process(self, request: str, context: dict = None) -> str:
        """Mock processing method."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"[{self.name}] Processed: {request[:50]}..."


class ResearchOrchestrator:
    """Comprehensive research orchestrator demonstrating ADK patterns.
    
    This orchestrator coordinates multiple specialized agents to handle
    complex research workflows including:
    - Literature search and retrieval
    - Data analysis and processing
    - Report generation and synthesis
    - Quality assurance and validation
    """
    
    def __init__(self):
        """Initialize the research orchestrator."""
        self.setup_logging()
        
        # Initialize core components
        self.registry = AgentRegistry(
            health_check_interval=300,
            enable_auto_health_checks=True
        )
        
        self.manager = AgentManager(
            registry=self.registry,
            enable_auto_scaling=True,
            scaling_check_interval=60
        )
        
        self.delegation_engine = DelegationEngine(
            registry=self.registry,
            manager=self.manager,
            max_concurrent_delegations=5
        )
        
        self.orchestrator = MasterOrchestrator(
            name="research_orchestrator",
            model="gemini-2.0-flash",
            description="Advanced research orchestrator for complex research workflows"
        )
        
        self.request_analyzer = AdvancedRequestAnalyzer()
        
        # Initialize specialized agents
        self._initialize_research_agents()
        
        logging.info("Initialized ResearchOrchestrator")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('research_orchestrator.log')
            ]
        )
    
    def _initialize_research_agents(self):
        """Initialize specialized research agents."""
        
        # Literature Search Agent
        literature_agent = MockAgent(
            name="literature_search_agent",
            description="Specialized agent for academic literature search and retrieval",
            capabilities=["search", "retrieval", "academic_databases"]
        )
        
        # Data Analysis Agent
        analysis_agent = MockAgent(
            name="data_analysis_agent", 
            description="Advanced data analysis and statistical processing agent",
            capabilities=["analysis", "statistics", "data_processing", "visualization"]
        )
        
        # Content Synthesis Agent
        synthesis_agent = MockAgent(
            name="content_synthesis_agent",
            description="Expert content synthesis and report generation agent",
            capabilities=["generation", "synthesis", "writing", "summarization"]
        )
        
        # Quality Assurance Agent
        qa_agent = MockAgent(
            name="quality_assurance_agent",
            description="Quality assurance and validation specialist",
            capabilities=["validation", "quality_control", "fact_checking"]
        )
        
        # Citation Management Agent
        citation_agent = MockAgent(
            name="citation_management_agent",
            description="Citation formatting and reference management specialist",
            capabilities=["citation", "formatting", "reference_management"]
        )
        
        # Register all agents
        agents = [literature_agent, analysis_agent, synthesis_agent, qa_agent, citation_agent]
        
        for agent in agents:
            self.registry.register_agent(agent, create_tool=True)
            self.orchestrator.register_agent(agent, as_tool=True)
        
        # Create specialized agent pools
        self.manager.create_agent_pool(
            name="research_pool",
            capability="search",
            min_agents=1,
            max_agents=3,
            scale_threshold=0.7
        )
        
        self.manager.create_agent_pool(
            name="analysis_pool", 
            capability="analysis",
            min_agents=1,
            max_agents=2,
            scale_threshold=0.8
        )
        
        logging.info(f"Registered {len(agents)} specialized research agents")
    
    async def conduct_research(
        self,
        research_query: str,
        research_type: str = "comprehensive",
        output_format: str = "report",
        priority: str = "normal"
    ) -> dict:
        """Conduct comprehensive research using the orchestrator.
        
        Args:
            research_query: The research question or topic
            research_type: Type of research (quick, standard, comprehensive)
            output_format: Desired output format (summary, report, presentation)
            priority: Research priority (low, normal, high, urgent)
            
        Returns:
            Dictionary containing research results and metadata
        """
        logging.info(f"Starting research: {research_query}")
        
        # Analyze the research request
        context = {
            'domain': 'research',
            'research_type': research_type,
            'output_format': output_format,
            'priority': priority
        }
        
        analysis = await self.request_analyzer.analyze_request(
            research_query,
            context=context,
            request_id=f"research_{asyncio.get_event_loop().time()}"
        )
        
        logging.info(f"Request analysis: {analysis.complexity.value} complexity, {len(analysis.required_capabilities)} capabilities")
        
        # Create delegation request
        delegation_priority = {
            'low': DelegationPriority.LOW,
            'normal': DelegationPriority.NORMAL, 
            'high': DelegationPriority.HIGH,
            'urgent': DelegationPriority.URGENT
        }.get(priority, DelegationPriority.NORMAL)
        
        delegation_request = DelegationRequest(
            request_id=analysis.request_id,
            original_request=research_query,
            required_capabilities=analysis.required_capabilities,
            context=context,
            priority=delegation_priority,
            max_delegation_depth=3,
            timeout_seconds=int(analysis.estimated_duration)
        )
        
        # Execute research workflow based on type
        if research_type == "quick":
            return await self._quick_research_workflow(delegation_request)
        elif research_type == "standard":
            return await self._standard_research_workflow(delegation_request)
        else:  # comprehensive
            return await self._comprehensive_research_workflow(delegation_request)
    
    async def _quick_research_workflow(self, request: DelegationRequest) -> dict:
        """Execute a quick research workflow."""
        logging.info("Executing quick research workflow")
        
        # Single agent delegation for quick research
        result = await self.delegation_engine.delegate_request(request)
        
        return {
            'workflow_type': 'quick',
            'research_results': result.result if result.success else "Research failed",
            'execution_time': result.execution_time,
            'success': result.success,
            'agent_used': result.selected_agent,
            'metadata': result.metadata
        }
    
    async def _standard_research_workflow(self, request: DelegationRequest) -> dict:
        """Execute a standard research workflow with sequential processing."""
        logging.info("Executing standard research workflow")
        
        workflow_results = []
        
        # Step 1: Literature search
        search_request = DelegationRequest(
            request_id=f"{request.request_id}_search",
            original_request=f"Search for literature on: {request.original_request}",
            required_capabilities=["search", "retrieval"],
            context=request.context,
            priority=request.priority
        )
        
        search_result = await self.delegation_engine.delegate_request(search_request)
        workflow_results.append(('literature_search', search_result))
        
        # Step 2: Analysis (if search was successful)
        if search_result.success:
            analysis_request = DelegationRequest(
                request_id=f"{request.request_id}_analysis",
                original_request=f"Analyze the research findings: {search_result.result}",
                required_capabilities=["analysis", "data_processing"],
                context=request.context,
                priority=request.priority
            )
            
            analysis_result = await self.delegation_engine.delegate_request(analysis_request)
            workflow_results.append(('analysis', analysis_result))
            
            # Step 3: Synthesis
            if analysis_result.success:
                synthesis_request = DelegationRequest(
                    request_id=f"{request.request_id}_synthesis",
                    original_request=f"Synthesize research into {request.context.get('output_format', 'report')}: {analysis_result.result}",
                    required_capabilities=["generation", "synthesis"],
                    context=request.context,
                    priority=request.priority
                )
                
                synthesis_result = await self.delegation_engine.delegate_request(synthesis_request)
                workflow_results.append(('synthesis', synthesis_result))
        
        # Compile results
        total_time = sum(result.execution_time for _, result in workflow_results)
        success = all(result.success for _, result in workflow_results)
        
        return {
            'workflow_type': 'standard',
            'workflow_steps': len(workflow_results),
            'research_results': workflow_results[-1][1].result if workflow_results and workflow_results[-1][1].success else "Workflow incomplete",
            'execution_time': total_time,
            'success': success,
            'step_results': [(step, result.success, result.execution_time) for step, result in workflow_results],
            'agents_used': [result.selected_agent for _, result in workflow_results if result.selected_agent]
        }
    
    async def _comprehensive_research_workflow(self, request: DelegationRequest) -> dict:
        """Execute a comprehensive research workflow with parallel and sequential processing."""
        logging.info("Executing comprehensive research workflow")
        
        workflow_results = []
        
        # Phase 1: Parallel literature search and initial analysis
        search_tasks = []
        
        # Multiple search strategies
        search_strategies = [
            ("academic_search", "Search academic databases and journals"),
            ("web_search", "Search web sources and recent publications"),
            ("expert_search", "Search for expert opinions and industry reports")
        ]
        
        for strategy_name, strategy_desc in search_strategies:
            search_request = DelegationRequest(
                request_id=f"{request.request_id}_{strategy_name}",
                original_request=f"{strategy_desc} for: {request.original_request}",
                required_capabilities=["search", "retrieval"],
                context={**request.context, 'search_strategy': strategy_name},
                priority=request.priority
            )
            
            task = self.delegation_engine.delegate_request(search_request)
            search_tasks.append((strategy_name, task))
        
        # Execute parallel searches
        search_results = []
        if search_tasks:
            results = await asyncio.gather(
                *[task for _, task in search_tasks],
                return_exceptions=True
            )
            
            for (strategy_name, _), result in zip(search_tasks, results):
                if not isinstance(result, Exception):
                    search_results.append((strategy_name, result))
        
        workflow_results.extend(search_results)
        
        # Phase 2: Comprehensive analysis
        if search_results:
            # Combine search results
            combined_findings = " | ".join([
                result.result for _, result in search_results if result.success
            ])
            
            analysis_request = DelegationRequest(
                request_id=f"{request.request_id}_comprehensive_analysis",
                original_request=f"Perform comprehensive analysis of research findings: {combined_findings}",
                required_capabilities=["analysis", "data_processing", "statistics"],
                context=request.context,
                priority=request.priority
            )
            
            analysis_result = await self.delegation_engine.delegate_request(analysis_request)
            workflow_results.append(('comprehensive_analysis', analysis_result))
            
            # Phase 3: Quality assurance and validation
            if analysis_result.success:
                qa_request = DelegationRequest(
                    request_id=f"{request.request_id}_quality_assurance",
                    original_request=f"Validate and quality check research analysis: {analysis_result.result}",
                    required_capabilities=["validation", "quality_control"],
                    context=request.context,
                    priority=request.priority
                )
                
                qa_result = await self.delegation_engine.delegate_request(qa_request)
                workflow_results.append(('quality_assurance', qa_result))
                
                # Phase 4: Final synthesis and formatting
                if qa_result.success:
                    synthesis_request = DelegationRequest(
                        request_id=f"{request.request_id}_final_synthesis",
                        original_request=f"Create final {request.context.get('output_format', 'report')} with citations: {qa_result.result}",
                        required_capabilities=["generation", "synthesis", "citation", "formatting"],
                        context=request.context,
                        priority=request.priority
                    )
                    
                    synthesis_result = await self.delegation_engine.delegate_request(synthesis_request)
                    workflow_results.append(('final_synthesis', synthesis_result))
        
        # Compile comprehensive results
        total_time = sum(result.execution_time for _, result in workflow_results if hasattr(result, 'execution_time'))
        success = any(result.success for _, result in workflow_results if hasattr(result, 'success'))
        
        return {
            'workflow_type': 'comprehensive',
            'workflow_phases': 4,
            'total_steps': len(workflow_results),
            'research_results': workflow_results[-1][1].result if workflow_results and hasattr(workflow_results[-1][1], 'result') else "Comprehensive research completed",
            'execution_time': total_time,
            'success': success,
            'phase_results': {
                'parallel_search': len([r for r in workflow_results if 'search' in r[0]]),
                'analysis_completed': any('analysis' in r[0] for r in workflow_results),
                'quality_assured': any('quality' in r[0] for r in workflow_results),
                'synthesis_completed': any('synthesis' in r[0] for r in workflow_results)
            },
            'agents_used': list(set([
                result.selected_agent for _, result in workflow_results 
                if hasattr(result, 'selected_agent') and result.selected_agent
            ]))
        }
    
    async def get_orchestrator_status(self) -> dict:
        """Get comprehensive status of the research orchestrator."""
        registry_stats = self.registry.get_registry_stats()
        manager_stats = self.manager.get_manager_stats()
        delegation_stats = self.delegation_engine.get_delegation_stats()
        orchestrator_status = self.orchestrator.get_orchestrator_status()
        
        return {
            'orchestrator': orchestrator_status,
            'registry': registry_stats,
            'manager': manager_stats,
            'delegation': delegation_stats,
            'health': {
                'total_agents': registry_stats['total_agents'],
                'active_agents': manager_stats['total_agents_managed'],
                'success_rate': delegation_stats['success_rate'],
                'avg_response_time': delegation_stats['average_execution_time']
            }
        }
    
    async def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        logging.info("Shutting down ResearchOrchestrator")
        
        await self.manager.shutdown()
        await self.registry.shutdown()
        
        logging.info("ResearchOrchestrator shutdown complete")


async def main():
    """Main function demonstrating the research orchestrator."""
    orchestrator = ResearchOrchestrator()
    
    try:
        # Example research queries
        research_queries = [
            {
                'query': "What are the latest developments in quantum computing for cryptography?",
                'type': "comprehensive",
                'format': "report",
                'priority': "high"
            },
            {
                'query': "How does machine learning impact healthcare diagnostics?",
                'type': "standard", 
                'format': "summary",
                'priority': "normal"
            },
            {
                'query': "What are the key benefits of renewable energy?",
                'type': "quick",
                'format': "summary",
                'priority': "low"
            }
        ]
        
        # Execute research queries
        for i, query_config in enumerate(research_queries, 1):
            print(f"\n{'='*60}")
            print(f"RESEARCH EXAMPLE {i}: {query_config['type'].upper()} RESEARCH")
            print(f"{'='*60}")
            print(f"Query: {query_config['query']}")
            print(f"Type: {query_config['type']}")
            print(f"Format: {query_config['format']}")
            print(f"Priority: {query_config['priority']}")
            print()
            
            # Conduct research
            result = await orchestrator.conduct_research(
                research_query=query_config['query'],
                research_type=query_config['type'],
                output_format=query_config['format'],
                priority=query_config['priority']
            )
            
            # Display results
            print("RESULTS:")
            print(f"Success: {result['success']}")
            print(f"Execution Time: {result['execution_time']:.2f}s")
            print(f"Workflow Type: {result['workflow_type']}")
            
            if 'agents_used' in result:
                print(f"Agents Used: {', '.join(result['agents_used'])}")
            
            if 'step_results' in result:
                print("Workflow Steps:")
                for step, success, time in result['step_results']:
                    print(f"  - {step}: {'✓' if success else '✗'} ({time:.2f}s)")
            
            print(f"Research Results: {result['research_results'][:200]}...")
            
            # Small delay between queries
            await asyncio.sleep(1)
        
        # Display orchestrator status
        print(f"\n{'='*60}")
        print("ORCHESTRATOR STATUS")
        print(f"{'='*60}")
        
        status = await orchestrator.get_orchestrator_status()
        print(f"Total Agents: {status['health']['total_agents']}")
        print(f"Active Agents: {status['health']['active_agents']}")
        print(f"Success Rate: {status['health']['success_rate']:.2%}")
        print(f"Avg Response Time: {status['health']['avg_response_time']:.2f}s")
        
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

