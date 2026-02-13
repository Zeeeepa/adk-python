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

"""Advanced request analysis for intelligent delegation."""

from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RequestType(str, Enum):
    """Types of requests that can be analyzed."""
    QUERY = "query"
    COMMAND = "command"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    TRANSFORMATION = "transformation"
    COMMUNICATION = "communication"
    WORKFLOW = "workflow"


class RequestComplexity(str, Enum):
    """Complexity levels for requests."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class RequestUrgency(str, Enum):
    """Urgency levels for requests."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class RequestIntent(BaseModel):
    """Analyzed intent of a request."""
    
    primary_intent: str = Field(description="Primary intent of the request")
    secondary_intents: List[str] = Field(default_factory=list, description="Secondary intents")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in intent analysis")
    reasoning: str = Field(description="Reasoning for the intent classification")


class RequestAnalysis(BaseModel):
    """Comprehensive analysis of a request."""
    
    request_id: str = Field(description="Unique request identifier")
    original_request: str = Field(description="Original request text")
    request_type: RequestType = Field(description="Type of request")
    complexity: RequestComplexity = Field(description="Complexity level")
    urgency: RequestUrgency = Field(description="Urgency level")
    intent: RequestIntent = Field(description="Analyzed intent")
    required_capabilities: List[str] = Field(description="Required agent capabilities")
    capability_scores: Dict[str, float] = Field(description="Confidence scores for capabilities")
    estimated_duration: float = Field(description="Estimated execution time in seconds")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    dependencies: List[str] = Field(default_factory=list, description="External dependencies")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    analyzed_at: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class AdvancedRequestAnalyzer:
    """Advanced request analyzer with sophisticated NLP and pattern recognition.
    
    Provides comprehensive analysis of user requests including:
    - Intent classification and confidence scoring
    - Complexity assessment with multiple factors
    - Capability requirement extraction
    - Resource and dependency analysis
    - Risk factor identification
    """
    
    def __init__(self):
        """Initialize the advanced request analyzer."""
        self._initialize_patterns()
        self._initialize_capability_mappings()
        self._initialize_complexity_indicators()
        
        logger.info("Initialized AdvancedRequestAnalyzer")
    
    def _initialize_patterns(self) -> None:
        """Initialize regex patterns for request analysis."""
        self.patterns = {
            'question_indicators': re.compile(
                r'\b(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does|did)\b',
                re.IGNORECASE
            ),
            'command_indicators': re.compile(
                r'\b(create|make|build|generate|write|send|delete|remove|update|modify|execute|run)\b',
                re.IGNORECASE
            ),
            'analysis_indicators': re.compile(
                r'\b(analyze|examine|study|evaluate|assess|compare|review|investigate|research)\b',
                re.IGNORECASE
            ),
            'urgency_indicators': re.compile(
                r'\b(urgent|asap|immediately|quickly|fast|rush|critical|emergency|now)\b',
                re.IGNORECASE
            ),
            'complexity_indicators': re.compile(
                r'\b(complex|complicated|detailed|comprehensive|thorough|advanced|sophisticated)\b',
                re.IGNORECASE
            ),
            'time_constraints': re.compile(
                r'\b(by|before|within|in|deadline|due|schedule)\s+(\d+\s*(minutes?|hours?|days?|weeks?))\b',
                re.IGNORECASE
            ),
            'data_references': re.compile(
                r'\b(data|dataset|database|file|document|report|spreadsheet|csv|json|xml)\b',
                re.IGNORECASE
            ),
        }
    
    def _initialize_capability_mappings(self) -> None:
        """Initialize capability mappings with expanded keyword sets."""
        self.capability_mappings = {
            'search': {
                'keywords': ['search', 'find', 'lookup', 'query', 'retrieve', 'locate', 'discover', 'seek'],
                'patterns': [r'\bfind\s+(?:me\s+)?(?:all\s+)?(.+)', r'\bsearch\s+for\s+(.+)', r'\blook\s+up\s+(.+)'],
                'weight': 1.0,
            },
            'analysis': {
                'keywords': ['analyze', 'examine', 'study', 'evaluate', 'assess', 'investigate', 'review', 'inspect'],
                'patterns': [r'\banalyze\s+(.+)', r'\bexamine\s+(.+)', r'\bwhat\s+does\s+(.+)\s+mean'],
                'weight': 1.2,
            },
            'generation': {
                'keywords': ['generate', 'create', 'write', 'produce', 'compose', 'draft', 'make', 'build'],
                'patterns': [r'\bcreate\s+(?:a\s+)?(.+)', r'\bwrite\s+(?:a\s+)?(.+)', r'\bgenerate\s+(.+)'],
                'weight': 1.1,
            },
            'processing': {
                'keywords': ['process', 'transform', 'convert', 'parse', 'extract', 'format', 'clean', 'normalize'],
                'patterns': [r'\bconvert\s+(.+)\s+to\s+(.+)', r'\bprocess\s+(.+)', r'\bextract\s+(.+)\s+from\s+(.+)'],
                'weight': 1.0,
            },
            'calculation': {
                'keywords': ['calculate', 'compute', 'sum', 'count', 'measure', 'quantify', 'estimate', 'determine'],
                'patterns': [r'\bcalculate\s+(.+)', r'\bcount\s+(.+)', r'\bhow\s+many\s+(.+)'],
                'weight': 0.9,
            },
            'communication': {
                'keywords': ['send', 'notify', 'message', 'email', 'alert', 'inform', 'contact', 'reach'],
                'patterns': [r'\bsend\s+(.+)\s+to\s+(.+)', r'\bnotify\s+(.+)', r'\bemail\s+(.+)'],
                'weight': 0.8,
            },
            'visualization': {
                'keywords': ['visualize', 'plot', 'chart', 'graph', 'display', 'show', 'render', 'draw'],
                'patterns': [r'\bplot\s+(.+)', r'\bchart\s+(.+)', r'\bvisualize\s+(.+)'],
                'weight': 0.9,
            },
            'integration': {
                'keywords': ['integrate', 'connect', 'sync', 'merge', 'combine', 'link', 'join', 'unite'],
                'patterns': [r'\bintegrate\s+(.+)\s+with\s+(.+)', r'\bconnect\s+(.+)\s+to\s+(.+)'],
                'weight': 1.1,
            },
            'monitoring': {
                'keywords': ['monitor', 'track', 'watch', 'observe', 'check', 'verify', 'validate', 'test'],
                'patterns': [r'\bmonitor\s+(.+)', r'\btrack\s+(.+)', r'\bcheck\s+if\s+(.+)'],
                'weight': 0.8,
            },
            'optimization': {
                'keywords': ['optimize', 'improve', 'enhance', 'tune', 'refine', 'streamline', 'boost', 'accelerate'],
                'patterns': [r'\boptimize\s+(.+)', r'\bimprove\s+(.+)', r'\bmake\s+(.+)\s+faster'],
                'weight': 1.2,
            },
        }
    
    def _initialize_complexity_indicators(self) -> None:
        """Initialize complexity assessment indicators."""
        self.complexity_indicators = {
            'trivial': {
                'keywords': ['simple', 'basic', 'quick', 'easy', 'straightforward'],
                'patterns': [r'\bjust\s+(.+)', r'\bsimply\s+(.+)', r'\bquickly\s+(.+)'],
                'max_capabilities': 1,
                'max_words': 10,
                'base_score': 0.1,
            },
            'simple': {
                'keywords': ['single', 'one', 'basic', 'standard', 'normal'],
                'patterns': [r'\bone\s+(.+)', r'\ba\s+single\s+(.+)'],
                'max_capabilities': 2,
                'max_words': 20,
                'base_score': 0.3,
            },
            'moderate': {
                'keywords': ['multiple', 'several', 'compare', 'analyze', 'process'],
                'patterns': [r'\bmultiple\s+(.+)', r'\bseveral\s+(.+)', r'\bcompare\s+(.+)\s+(?:and|with)\s+(.+)'],
                'max_capabilities': 3,
                'max_words': 40,
                'base_score': 0.5,
            },
            'complex': {
                'keywords': ['comprehensive', 'detailed', 'advanced', 'sophisticated', 'complex'],
                'patterns': [r'\bcomprehensive\s+(.+)', r'\bdetailed\s+(.+)', r'\badvanced\s+(.+)'],
                'max_capabilities': 5,
                'max_words': 80,
                'base_score': 0.7,
            },
            'expert': {
                'keywords': ['expert', 'professional', 'enterprise', 'production', 'critical'],
                'patterns': [r'\bexpert\s+(.+)', r'\bprofessional\s+(.+)', r'\benterprise\s+(.+)'],
                'max_capabilities': 10,
                'max_words': 150,
                'base_score': 0.9,
            },
        }
    
    async def analyze_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> RequestAnalysis:
        """Perform comprehensive analysis of a request.
        
        Args:
            request: The request text to analyze
            context: Optional context information
            request_id: Optional request identifier
            
        Returns:
            Comprehensive request analysis
        """
        context = context or {}
        request_id = request_id or f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.debug(f"Analyzing request {request_id}: {request[:100]}...")
        
        # Analyze request type
        request_type = self._classify_request_type(request)
        
        # Analyze complexity
        complexity = self._assess_complexity(request, context)
        
        # Analyze urgency
        urgency = self._assess_urgency(request, context)
        
        # Analyze intent
        intent = self._analyze_intent(request, context)
        
        # Extract capabilities
        capabilities, capability_scores = self._extract_capabilities(request, context)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(request, complexity, capabilities)
        
        # Assess resource requirements
        resource_requirements = self._assess_resource_requirements(request, capabilities)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(request, complexity, capabilities)
        
        # Identify dependencies
        dependencies = self._identify_dependencies(request, context)
        
        analysis = RequestAnalysis(
            request_id=request_id,
            original_request=request,
            request_type=request_type,
            complexity=complexity,
            urgency=urgency,
            intent=intent,
            required_capabilities=capabilities,
            capability_scores=capability_scores,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements,
            risk_factors=risk_factors,
            dependencies=dependencies,
            metadata={
                'word_count': len(request.split()),
                'character_count': len(request),
                'has_time_constraints': bool(self.patterns['time_constraints'].search(request)),
                'has_data_references': bool(self.patterns['data_references'].search(request)),
                'context_provided': bool(context),
            },
        )
        
        logger.info(f"Completed analysis for request {request_id}: {complexity.value} complexity, {len(capabilities)} capabilities")
        return analysis
    
    def _classify_request_type(self, request: str) -> RequestType:
        """Classify the type of request."""
        request_lower = request.lower()
        
        # Check for question indicators
        if self.patterns['question_indicators'].search(request) or request.strip().endswith('?'):
            return RequestType.QUERY
        
        # Check for command indicators
        if self.patterns['command_indicators'].search(request):
            return RequestType.COMMAND
        
        # Check for analysis indicators
        if self.patterns['analysis_indicators'].search(request):
            return RequestType.ANALYSIS
        
        # Check for generation keywords
        generation_keywords = ['create', 'generate', 'write', 'produce', 'compose', 'draft']
        if any(keyword in request_lower for keyword in generation_keywords):
            return RequestType.GENERATION
        
        # Check for transformation keywords
        transform_keywords = ['convert', 'transform', 'process', 'parse', 'extract', 'format']
        if any(keyword in request_lower for keyword in transform_keywords):
            return RequestType.TRANSFORMATION
        
        # Check for communication keywords
        comm_keywords = ['send', 'notify', 'message', 'email', 'alert', 'inform']
        if any(keyword in request_lower for keyword in comm_keywords):
            return RequestType.COMMUNICATION
        
        # Check for workflow keywords
        workflow_keywords = ['workflow', 'process', 'pipeline', 'sequence', 'steps']
        if any(keyword in request_lower for keyword in workflow_keywords):
            return RequestType.WORKFLOW
        
        # Default to query if unclear
        return RequestType.QUERY
    
    def _assess_complexity(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> RequestComplexity:
        """Assess the complexity of the request."""
        request_lower = request.lower()
        word_count = len(request.split())
        
        complexity_scores = {}
        
        # Score based on indicators
        for complexity_level, indicators in self.complexity_indicators.items():
            score = indicators['base_score']
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in indicators['keywords'] if keyword in request_lower)
            if keyword_matches > 0:
                score += keyword_matches * 0.1
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in indicators['patterns'] if re.search(pattern, request, re.IGNORECASE))
            if pattern_matches > 0:
                score += pattern_matches * 0.15
            
            # Word count factor
            if word_count <= indicators['max_words']:
                score += 0.1
            
            complexity_scores[complexity_level] = score
        
        # Additional complexity factors
        if self.patterns['complexity_indicators'].search(request):
            complexity_scores['complex'] = complexity_scores.get('complex', 0) + 0.2
            complexity_scores['expert'] = complexity_scores.get('expert', 0) + 0.1
        
        # Context-based adjustments
        if context.get('domain') in ['research', 'analysis', 'enterprise']:
            complexity_scores['complex'] = complexity_scores.get('complex', 0) + 0.1
            complexity_scores['expert'] = complexity_scores.get('expert', 0) + 0.1
        
        # Select highest scoring complexity
        max_complexity = max(complexity_scores, key=complexity_scores.get)
        return RequestComplexity(max_complexity)
    
    def _assess_urgency(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> RequestUrgency:
        """Assess the urgency of the request."""
        request_lower = request.lower()
        
        # Check for explicit urgency indicators
        if self.patterns['urgency_indicators'].search(request):
            if any(word in request_lower for word in ['critical', 'emergency', 'urgent']):
                return RequestUrgency.CRITICAL
            elif any(word in request_lower for word in ['asap', 'immediately', 'quickly']):
                return RequestUrgency.URGENT
            else:
                return RequestUrgency.HIGH
        
        # Check for time constraints
        time_match = self.patterns['time_constraints'].search(request)
        if time_match:
            time_text = time_match.group(2).lower()
            if 'minute' in time_text:
                return RequestUrgency.URGENT
            elif 'hour' in time_text:
                return RequestUrgency.HIGH
            elif 'day' in time_text:
                return RequestUrgency.NORMAL
        
        # Context-based urgency
        if context.get('priority') == 'high':
            return RequestUrgency.HIGH
        elif context.get('priority') == 'critical':
            return RequestUrgency.CRITICAL
        
        return RequestUrgency.NORMAL
    
    def _analyze_intent(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> RequestIntent:
        """Analyze the intent of the request."""
        request_lower = request.lower()
        
        # Primary intent detection
        intent_scores = {}
        
        # Information seeking
        if any(word in request_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            intent_scores['information_seeking'] = 0.8
        
        # Task execution
        if any(word in request_lower for word in ['create', 'make', 'build', 'generate', 'do']):
            intent_scores['task_execution'] = 0.9
        
        # Problem solving
        if any(word in request_lower for word in ['solve', 'fix', 'resolve', 'troubleshoot', 'debug']):
            intent_scores['problem_solving'] = 0.9
        
        # Analysis and evaluation
        if any(word in request_lower for word in ['analyze', 'evaluate', 'assess', 'compare', 'review']):
            intent_scores['analysis'] = 0.8
        
        # Learning and explanation
        if any(word in request_lower for word in ['explain', 'teach', 'learn', 'understand', 'clarify']):
            intent_scores['learning'] = 0.7
        
        # Default intent
        if not intent_scores:
            intent_scores['general_assistance'] = 0.5
        
        # Select primary intent
        primary_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[primary_intent]
        
        # Secondary intents (with lower scores)
        secondary_intents = [
            intent for intent, score in intent_scores.items()
            if intent != primary_intent and score >= 0.3
        ]
        
        reasoning = f"Detected '{primary_intent}' based on keyword analysis and request structure"
        
        return RequestIntent(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            reasoning=reasoning,
        )
    
    def _extract_capabilities(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> Tuple[List[str], Dict[str, float]]:
        """Extract required capabilities from the request."""
        request_lower = request.lower()
        capability_scores = {}
        
        # Keyword-based capability detection
        for capability, mapping in self.capability_mappings.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in mapping['keywords'] if keyword in request_lower)
            if keyword_matches > 0:
                score += (keyword_matches / len(mapping['keywords'])) * mapping['weight']
            
            # Check patterns
            pattern_matches = sum(1 for pattern in mapping['patterns'] if re.search(pattern, request, re.IGNORECASE))
            if pattern_matches > 0:
                score += (pattern_matches * 0.2) * mapping['weight']
            
            if score > 0:
                capability_scores[capability] = min(score, 1.0)
        
        # Context-based capability enhancement
        if context.get('domain'):
            domain = context['domain'].lower()
            if domain == 'research':
                capability_scores['search'] = capability_scores.get('search', 0.0) + 0.3
                capability_scores['analysis'] = capability_scores.get('analysis', 0.0) + 0.2
            elif domain == 'data':
                capability_scores['processing'] = capability_scores.get('processing', 0.0) + 0.3
                capability_scores['analysis'] = capability_scores.get('analysis', 0.0) + 0.3
            elif domain == 'content':
                capability_scores['generation'] = capability_scores.get('generation', 0.0) + 0.3
        
        # Filter capabilities with minimum confidence
        min_confidence = 0.1
        filtered_capabilities = {
            cap: score for cap, score in capability_scores.items()
            if score >= min_confidence
        }
        
        # Ensure at least one capability
        if not filtered_capabilities:
            filtered_capabilities['general'] = 0.5
        
        # Sort by confidence
        sorted_capabilities = sorted(filtered_capabilities.items(), key=lambda x: x[1], reverse=True)
        required_capabilities = [cap for cap, _ in sorted_capabilities]
        
        return required_capabilities, filtered_capabilities
    
    def _estimate_duration(
        self,
        request: str,
        complexity: RequestComplexity,
        capabilities: List[str],
    ) -> float:
        """Estimate the duration for request execution."""
        base_durations = {
            RequestComplexity.TRIVIAL: 10.0,
            RequestComplexity.SIMPLE: 30.0,
            RequestComplexity.MODERATE: 120.0,
            RequestComplexity.COMPLEX: 300.0,
            RequestComplexity.EXPERT: 600.0,
        }
        
        base_duration = base_durations[complexity]
        
        # Adjust based on number of capabilities
        capability_multiplier = 1.0 + (len(capabilities) - 1) * 0.2
        
        # Adjust based on request length
        word_count = len(request.split())
        length_multiplier = 1.0 + (word_count / 100.0) * 0.1
        
        estimated_duration = base_duration * capability_multiplier * length_multiplier
        
        return min(estimated_duration, 3600.0)  # Cap at 1 hour
    
    def _assess_resource_requirements(
        self,
        request: str,
        capabilities: List[str],
    ) -> Dict[str, Any]:
        """Assess resource requirements for the request."""
        requirements = {
            'cpu_intensive': False,
            'memory_intensive': False,
            'network_intensive': False,
            'storage_intensive': False,
            'external_apis': False,
        }
        
        request_lower = request.lower()
        
        # CPU intensive operations
        if any(cap in capabilities for cap in ['analysis', 'processing', 'calculation', 'optimization']):
            requirements['cpu_intensive'] = True
        
        # Memory intensive operations
        if any(word in request_lower for word in ['large', 'big', 'massive', 'dataset', 'bulk']):
            requirements['memory_intensive'] = True
        
        # Network intensive operations
        if any(cap in capabilities for cap in ['search', 'communication', 'integration']):
            requirements['network_intensive'] = True
        
        # Storage intensive operations
        if any(word in request_lower for word in ['file', 'document', 'data', 'store', 'save']):
            requirements['storage_intensive'] = True
        
        # External API requirements
        if any(word in request_lower for word in ['api', 'service', 'external', 'third-party']):
            requirements['external_apis'] = True
        
        return requirements
    
    def _identify_risk_factors(
        self,
        request: str,
        complexity: RequestComplexity,
        capabilities: List[str],
    ) -> List[str]:
        """Identify potential risk factors in the request."""
        risk_factors = []
        request_lower = request.lower()
        
        # Complexity-based risks
        if complexity in [RequestComplexity.COMPLEX, RequestComplexity.EXPERT]:
            risk_factors.append("high_complexity")
        
        # Multiple capability risks
        if len(capabilities) > 3:
            risk_factors.append("multiple_capabilities")
        
        # Data handling risks
        if any(word in request_lower for word in ['sensitive', 'confidential', 'private', 'personal']):
            risk_factors.append("sensitive_data")
        
        # External dependency risks
        if any(word in request_lower for word in ['external', 'third-party', 'api', 'service']):
            risk_factors.append("external_dependencies")
        
        # Time constraint risks
        if any(word in request_lower for word in ['urgent', 'asap', 'immediately', 'deadline']):
            risk_factors.append("time_constraints")
        
        # Scale risks
        if any(word in request_lower for word in ['large', 'massive', 'bulk', 'thousands', 'millions']):
            risk_factors.append("scale_concerns")
        
        return risk_factors
    
    def _identify_dependencies(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> List[str]:
        """Identify external dependencies for the request."""
        dependencies = []
        request_lower = request.lower()
        
        # Data source dependencies
        if any(word in request_lower for word in ['database', 'file', 'document', 'spreadsheet']):
            dependencies.append("data_sources")
        
        # API dependencies
        if any(word in request_lower for word in ['api', 'service', 'endpoint']):
            dependencies.append("external_apis")
        
        # Authentication dependencies
        if any(word in request_lower for word in ['login', 'auth', 'credential', 'token']):
            dependencies.append("authentication")
        
        # Network dependencies
        if any(word in request_lower for word in ['internet', 'network', 'online', 'web']):
            dependencies.append("network_connectivity")
        
        # Context-based dependencies
        if context.get('requires_approval'):
            dependencies.append("approval_workflow")
        
        if context.get('user_permissions'):
            dependencies.append("user_permissions")
        
        return dependencies

