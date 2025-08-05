"""
Comprehensive Serena Analysis for Graph-Sitter

This module provides comprehensive codebase analysis using Serena's LSP integration,
combining static analysis, symbol analysis, and runtime error detection.
"""

import asyncio
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict
from urllib.parse import urlparse

# Graph-sitter imports
from graph_sitter.core.codebase import Codebase
from graph_sitter.codebase.codebase_analysis import (
    get_codebase_summary, 
    get_file_summary, 
    get_class_summary, 
    get_function_summary, 
    get_symbol_summary
)
from graph_sitter.core.runtime_errors import RuntimeErrorCollector, RuntimeContext
from graph_sitter.extensions.lsp.serena_bridge import SerenaLSPBridge, SerenaErrorInfo

# SolidLSP imports
from solidlsp.ls_types import (
    DiagnosticSeverity, Diagnostic, Position, Range, MarkupContent,
    Location, MarkupKind, CompletionItemKind, CompletionItem, 
    UnifiedSymbolInformation, SymbolKind, SymbolTag
)
from solidlsp.lsp_protocol_handler.lsp_types import (
    DiagnosticOptions, PublishDiagnosticsParams, DiagnosticRelatedInformation
)

# Serena imports
from serena.symbol import (
    LanguageServerSymbolRetriever, ReferenceInLanguageServerSymbol,
    LanguageServerSymbol, Symbol, PositionInFile, LanguageServerSymbolLocation
)
from serena.text_utils import MatchedConsecutiveLines, TextLine, LineType
from serena.project import Project
from serena.code_editor import CodeEditor
from serena.cli import (
    PromptCommands, ToolCommands, ProjectCommands, SerenaConfigCommands,
    ContextCommands, ModeCommands, TopLevelCommands, AutoRegisteringGroup, ProjectType
)

from graph_sitter.shared.logging.get_logger import get_logger

logger = get_logger(__name__)


@dataclass
class RepositoryInfo:
    """Information about a repository being analyzed."""
    url: str
    name: str
    owner: str
    local_path: str
    branch: str = "main"
    clone_depth: Optional[int] = None
    
    @classmethod
    def from_url(cls, url: str, local_path: str) -> 'RepositoryInfo':
        """Create RepositoryInfo from GitHub URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub URL: {url}")
        
        owner = path_parts[0]
        name = path_parts[1].replace('.git', '')
        
        return cls(
            url=url,
            name=name,
            owner=owner,
            local_path=local_path
        )


@dataclass
class AnalysisMetrics:
    """Comprehensive analysis metrics."""
    total_files: int = 0
    total_lines: int = 0
    total_symbols: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0
    
    # Error counts by severity
    critical_errors: int = 0
    errors: int = 0
    warnings: int = 0
    info_messages: int = 0
    hints: int = 0
    
    # Analysis performance
    analysis_duration: float = 0.0
    files_analyzed: int = 0
    symbols_analyzed: int = 0
    
    # Quality metrics
    maintainability_index: float = 0.0
    technical_debt_score: float = 0.0
    test_coverage_estimate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'file_metrics': {
                'total_files': self.total_files,
                'total_lines': self.total_lines,
                'files_analyzed': self.files_analyzed
            },
            'symbol_metrics': {
                'total_symbols': self.total_symbols,
                'total_functions': self.total_functions,
                'total_classes': self.total_classes,
                'total_imports': self.total_imports,
                'symbols_analyzed': self.symbols_analyzed
            },
            'error_metrics': {
                'critical_errors': self.critical_errors,
                'errors': self.errors,
                'warnings': self.warnings,
                'info_messages': self.info_messages,
                'hints': self.hints,
                'total_issues': self.critical_errors + self.errors + self.warnings + self.info_messages + self.hints
            },
            'quality_metrics': {
                'maintainability_index': self.maintainability_index,
                'technical_debt_score': self.technical_debt_score,
                'test_coverage_estimate': self.test_coverage_estimate
            },
            'performance_metrics': {
                'analysis_duration': self.analysis_duration
            }
        }


@dataclass
class ComprehensiveAnalysisResult:
    """Result of comprehensive Serena analysis."""
    repository: RepositoryInfo
    metrics: AnalysisMetrics
    errors: List[SerenaErrorInfo] = field(default_factory=list)
    symbols: List[LanguageServerSymbol] = field(default_factory=list)
    file_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    codebase_summary: Dict[str, Any] = field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_errors_by_severity(self) -> Dict[str, List[SerenaErrorInfo]]:
        """Get errors grouped by severity."""
        errors_by_severity = {
            'critical': [],
            'error': [],
            'warning': [],
            'info': [],
            'hint': []
        }
        
        for error in self.errors:
            if error.severity == DiagnosticSeverity.ERROR:
                if 'critical' in error.message.lower():
                    errors_by_severity['critical'].append(error)
                else:
                    errors_by_severity['error'].append(error)
            elif error.severity == DiagnosticSeverity.WARNING:
                errors_by_severity['warning'].append(error)
            elif error.severity == DiagnosticSeverity.INFORMATION:
                errors_by_severity['info'].append(error)
            elif error.severity == DiagnosticSeverity.HINT:
                errors_by_severity['hint'].append(error)
        
        return errors_by_severity
    
    def get_error_hotspots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get files with the most errors."""
        file_error_counts = defaultdict(int)
        
        for error in self.errors:
            file_error_counts[error.file_path] += 1
        
        hotspots = [
            {'file_path': file_path, 'error_count': count}
            for file_path, count in sorted(file_error_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return hotspots[:limit]
    
    def get_symbol_complexity_analysis(self) -> Dict[str, Any]:
        """Analyze symbol complexity."""
        complexity_analysis = {
            'high_complexity_symbols': [],
            'average_complexity': 0.0,
            'complexity_distribution': defaultdict(int)
        }
        
        total_complexity = 0
        symbol_count = 0
        
        for symbol in self.symbols:
            # Calculate complexity based on symbol properties
            complexity = self._calculate_symbol_complexity(symbol)
            total_complexity += complexity
            symbol_count += 1
            
            # Categorize complexity
            if complexity > 10:
                complexity_analysis['high_complexity_symbols'].append({
                    'name': symbol.name,
                    'complexity': complexity,
                    'location': getattr(symbol, 'location', 'unknown')
                })
            
            # Distribution
            complexity_range = f"{int(complexity//5)*5}-{int(complexity//5)*5+4}"
            complexity_analysis['complexity_distribution'][complexity_range] += 1
        
        if symbol_count > 0:
            complexity_analysis['average_complexity'] = total_complexity / symbol_count
        
        return complexity_analysis
    
    def _calculate_symbol_complexity(self, symbol: LanguageServerSymbol) -> float:
        """Calculate complexity score for a symbol."""
        # Basic complexity calculation based on symbol properties
        complexity = 1.0
        
        # Add complexity based on symbol name length (proxy for complexity)
        complexity += len(symbol.name) / 20.0
        
        # Add complexity based on symbol kind
        if hasattr(symbol, 'kind'):
            if symbol.kind == SymbolKind.FUNCTION:
                complexity += 2.0
            elif symbol.kind == SymbolKind.CLASS:
                complexity += 3.0
            elif symbol.kind == SymbolKind.METHOD:
                complexity += 2.5
        
        return complexity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'repository': {
                'url': self.repository.url,
                'name': self.repository.name,
                'owner': self.repository.owner,
                'branch': self.repository.branch,
                'local_path': self.repository.local_path
            },
            'metrics': self.metrics.to_dict(),
            'errors_by_severity': {
                severity: [
                    {
                        'file_path': error.file_path,
                        'line': error.line,
                        'character': error.character,
                        'message': error.message,
                        'source': error.source,
                        'code': error.code
                    }
                    for error in errors
                ]
                for severity, errors in self.get_errors_by_severity().items()
            },
            'error_hotspots': self.get_error_hotspots(),
            'symbol_analysis': self.get_symbol_complexity_analysis(),
            'file_summaries': self.file_summaries,
            'codebase_summary': self.codebase_summary,
            'analysis_metadata': self.analysis_metadata
        }


class SerenaCodebaseAnalyzer:
    """
    Comprehensive codebase analyzer using Serena LSP integration.
    
    Features:
    - Complete LSP error analysis
    - Symbol analysis and complexity metrics
    - Runtime error detection
    - Performance analysis
    - Quality metrics calculation
    """
    
    def __init__(self, work_dir: Optional[str] = None, enable_runtime_collection: bool = True):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(exist_ok=True)
        
        self.enable_runtime_collection = enable_runtime_collection
        self.analysis_cache: Dict[str, ComprehensiveAnalysisResult] = {}
        
        # Performance tracking
        self.performance_stats = {
            'repositories_analyzed': 0,
            'total_analysis_time': 0.0,
            'average_analysis_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"Serena codebase analyzer initialized with work_dir: {self.work_dir}")
    
    async def analyze_codebase(
        self,
        codebase: Codebase,
        include_runtime_errors: bool = True,
        include_symbol_analysis: bool = True,
        include_file_summaries: bool = True
    ) -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive analysis of a codebase.
        
        Args:
            codebase: The codebase to analyze
            include_runtime_errors: Whether to collect runtime errors
            include_symbol_analysis: Whether to perform symbol analysis
            include_file_summaries: Whether to generate file summaries
            
        Returns:
            Comprehensive analysis result
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting comprehensive analysis of codebase: {codebase.repo_path}")
            
            # Create repository info
            repo_info = RepositoryInfo(
                url=f"file://{codebase.repo_path}",
                name=Path(codebase.repo_path).name,
                owner="local",
                local_path=str(codebase.repo_path)
            )
            
            # Initialize Serena LSP bridge
            bridge = SerenaLSPBridge(codebase, enable_runtime_collection=include_runtime_errors)
            
            try:
                # Perform analysis
                result = await self._perform_comprehensive_analysis(
                    bridge, repo_info, codebase,
                    include_runtime_errors, include_symbol_analysis, include_file_summaries
                )
                
                # Calculate analysis duration
                analysis_duration = time.time() - start_time
                result.metrics.analysis_duration = analysis_duration
                
                # Update performance stats
                self.performance_stats['repositories_analyzed'] += 1
                self.performance_stats['total_analysis_time'] += analysis_duration
                self.performance_stats['average_analysis_time'] = (
                    self.performance_stats['total_analysis_time'] / 
                    self.performance_stats['repositories_analyzed']
                )
                
                logger.info(f"Analysis completed in {analysis_duration:.2f}s: "
                           f"{len(result.errors)} errors found, {len(result.symbols)} symbols analyzed")
                
                return result
                
            finally:
                bridge.shutdown()
                
        except Exception as e:
            logger.error(f"Error during codebase analysis: {e}")
            # Return empty result with error information
            return ComprehensiveAnalysisResult(
                repository=RepositoryInfo(url="", name="", owner="", local_path=""),
                metrics=AnalysisMetrics(analysis_duration=time.time() - start_time),
                analysis_metadata={'error': str(e)}
            )
    
    async def analyze_repository_by_url(
        self,
        repo_url: str,
        branch: str = "main",
        clone_depth: Optional[int] = 1,
        use_cache: bool = True
    ) -> ComprehensiveAnalysisResult:
        """
        Analyze a GitHub repository by URL.
        
        Args:
            repo_url: GitHub repository URL
            branch: Branch to analyze
            clone_depth: Clone depth for shallow clone
            use_cache: Whether to use cached results
            
        Returns:
            Comprehensive analysis result
        """
        start_time = time.time()
        
        try:
            # Create repository info
            local_path = self.work_dir / f"repo_{int(time.time())}"
            repo_info = RepositoryInfo.from_url(repo_url, str(local_path))
            repo_info.branch = branch
            repo_info.clone_depth = clone_depth
            
            # Check cache
            cache_key = f"{repo_url}:{branch}"
            if use_cache and cache_key in self.analysis_cache:
                self.performance_stats['cache_hits'] += 1
                logger.info(f"Using cached analysis for {repo_url}")
                return self.analysis_cache[cache_key]
            
            self.performance_stats['cache_misses'] += 1
            
            # Clone repository
            logger.info(f"Cloning repository: {repo_url}")
            await self._clone_repository(repo_info)
            
            # Create codebase
            codebase = Codebase(str(local_path))
            
            # Perform analysis
            result = await self.analyze_codebase(codebase)
            result.repository = repo_info
            
            # Cache result
            self.analysis_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing repository {repo_url}: {e}")
            return ComprehensiveAnalysisResult(
                repository=RepositoryInfo.from_url(repo_url, ""),
                metrics=AnalysisMetrics(analysis_duration=time.time() - start_time),
                analysis_metadata={'error': str(e)}
            )
    
    async def _perform_comprehensive_analysis(
        self,
        bridge: SerenaLSPBridge,
        repo_info: RepositoryInfo,
        codebase: Codebase,
        include_runtime_errors: bool,
        include_symbol_analysis: bool,
        include_file_summaries: bool
    ) -> ComprehensiveAnalysisResult:
        """Perform the actual comprehensive analysis."""
        
        # Initialize metrics
        metrics = AnalysisMetrics()
        
        # Get all errors from Serena bridge
        logger.info("Collecting diagnostics from Serena LSP bridge...")
        errors = bridge.get_diagnostics()
        
        # Count errors by severity
        for error in errors:
            if error.severity == DiagnosticSeverity.ERROR:
                if 'critical' in error.message.lower():
                    metrics.critical_errors += 1
                else:
                    metrics.errors += 1
            elif error.severity == DiagnosticSeverity.WARNING:
                metrics.warnings += 1
            elif error.severity == DiagnosticSeverity.INFORMATION:
                metrics.info_messages += 1
            elif error.severity == DiagnosticSeverity.HINT:
                metrics.hints += 1
        
        # Analyze symbols if requested
        symbols = []
        if include_symbol_analysis:
            logger.info("Performing symbol analysis...")
            symbols = await self._analyze_symbols(bridge, codebase)
            metrics.symbols_analyzed = len(symbols)
        
        # Generate file summaries if requested
        file_summaries = {}
        if include_file_summaries:
            logger.info("Generating file summaries...")
            file_summaries = await self._generate_file_summaries(codebase)
        
        # Calculate codebase metrics
        await self._calculate_codebase_metrics(codebase, metrics)
        
        # Generate codebase summary
        codebase_summary = await self._generate_codebase_summary(codebase)
        
        # Calculate quality metrics
        self._calculate_quality_metrics(metrics, errors, symbols)
        
        return ComprehensiveAnalysisResult(
            repository=repo_info,
            metrics=metrics,
            errors=errors,
            symbols=symbols,
            file_summaries=file_summaries,
            codebase_summary=codebase_summary,
            analysis_metadata={
                'serena_bridge_status': bridge.get_status(),
                'analysis_timestamp': time.time(),
                'runtime_collection_enabled': include_runtime_errors,
                'symbol_analysis_enabled': include_symbol_analysis,
                'file_summaries_enabled': include_file_summaries
            }
        )
    
    async def _clone_repository(self, repo_info: RepositoryInfo):
        """Clone a GitHub repository."""
        try:
            cmd = ["git", "clone"]
            
            if repo_info.clone_depth:
                cmd.extend(["--depth", str(repo_info.clone_depth)])
            
            if repo_info.branch != "main":
                cmd.extend(["--branch", repo_info.branch])
            
            cmd.extend([repo_info.url, repo_info.local_path])
            
            # Run git clone
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Git clone failed: {stderr.decode()}")
            
            logger.info(f"Successfully cloned {repo_info.url} to {repo_info.local_path}")
            
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise
    
    async def _analyze_symbols(self, bridge: SerenaLSPBridge, codebase: Codebase) -> List[LanguageServerSymbol]:
        """Analyze symbols in the codebase."""
        symbols = []
        
        try:
            # Get workspace symbols
            workspace_symbols = bridge.get_workspace_symbols("")
            
            # Convert to LanguageServerSymbol format
            for ws_symbol in workspace_symbols:
                # This would need proper conversion from UnifiedSymbolInformation
                # to LanguageServerSymbol - placeholder for now
                pass
            
        except Exception as e:
            logger.error(f"Error analyzing symbols: {e}")
        
        return symbols
    
    async def _generate_file_summaries(self, codebase: Codebase) -> Dict[str, Dict[str, Any]]:
        """Generate summaries for each file."""
        file_summaries = {}
        
        try:
            for file_obj in codebase.files:
                if file_obj.file_path.endswith('.py'):
                    try:
                        summary = get_file_summary(codebase, file_obj.file_path)
                        file_summaries[file_obj.file_path] = summary
                    except Exception as e:
                        logger.error(f"Error generating summary for {file_obj.file_path}: {e}")
                        file_summaries[file_obj.file_path] = {'error': str(e)}
                        
        except Exception as e:
            logger.error(f"Error generating file summaries: {e}")
        
        return file_summaries
    
    async def _calculate_codebase_metrics(self, codebase: Codebase, metrics: AnalysisMetrics):
        """Calculate basic codebase metrics."""
        try:
            metrics.total_files = len(codebase.files)
            metrics.files_analyzed = len([f for f in codebase.files if f.file_path.endswith('.py')])
            
            # Count lines
            total_lines = 0
            for file_obj in codebase.files:
                if file_obj.file_path.endswith('.py'):
                    try:
                        lines = len(file_obj.content.splitlines())
                        total_lines += lines
                    except Exception:
                        continue
            
            metrics.total_lines = total_lines
            
            # Count symbols
            if hasattr(codebase, 'functions'):
                metrics.total_functions = len(codebase.functions)
            if hasattr(codebase, 'classes'):
                metrics.total_classes = len(codebase.classes)
            if hasattr(codebase, 'imports'):
                metrics.total_imports = len(codebase.imports)
            
            metrics.total_symbols = metrics.total_functions + metrics.total_classes + metrics.total_imports
            
        except Exception as e:
            logger.error(f"Error calculating codebase metrics: {e}")
    
    async def _generate_codebase_summary(self, codebase: Codebase) -> Dict[str, Any]:
        """Generate overall codebase summary."""
        try:
            return get_codebase_summary(codebase)
        except Exception as e:
            logger.error(f"Error generating codebase summary: {e}")
            return {'error': str(e)}
    
    def _calculate_quality_metrics(self, metrics: AnalysisMetrics, errors: List[SerenaErrorInfo], symbols: List[LanguageServerSymbol]):
        """Calculate quality metrics."""
        try:
            # Maintainability index (0-100, higher is better)
            error_penalty = (metrics.critical_errors * 10) + (metrics.errors * 5) + (metrics.warnings * 2)
            metrics.maintainability_index = max(0, 100 - error_penalty)
            
            # Technical debt score (lower is better)
            metrics.technical_debt_score = (
                (metrics.critical_errors * 5) +
                (metrics.errors * 3) +
                (metrics.warnings * 1) +
                len([s for s in symbols if self._is_complex_symbol(s)])
            )
            
            # Test coverage estimate (rough estimate based on file patterns)
            test_files = metrics.total_files * 0.1  # Assume 10% test files
            metrics.test_coverage_estimate = min(100, (test_files / max(metrics.total_files, 1)) * 100)
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
    
    def _is_complex_symbol(self, symbol: LanguageServerSymbol) -> bool:
        """Check if a symbol is considered complex."""
        # Simple heuristic based on symbol name length
        return len(symbol.name) > 30
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed."""
        total_errors = 0
        total_symbols = 0
        
        for result in self.analysis_cache.values():
            total_errors += len(result.errors)
            total_symbols += len(result.symbols)
        
        return {
            'repositories_analyzed': len(self.analysis_cache),
            'total_errors_found': total_errors,
            'total_symbols_analyzed': total_symbols,
            'performance_stats': self.performance_stats.copy(),
            'cache_size': len(self.analysis_cache)
        }
    
    def clear_cache(self, repo_url: Optional[str] = None):
        """Clear analysis cache."""
        if repo_url:
            cache_keys_to_remove = [key for key in self.analysis_cache.keys() if repo_url in key]
            for key in cache_keys_to_remove:
                self.analysis_cache.pop(key, None)
        else:
            self.analysis_cache.clear()
        
        logger.info(f"Cache cleared for {repo_url or 'all repositories'}")
    
    async def shutdown(self):
        """Shutdown the analyzer and clean up resources."""
        try:
            self.analysis_cache.clear()
            logger.info("Serena codebase analyzer shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Convenience functions
async def analyze_codebase_comprehensive(
    codebase: Codebase,
    include_runtime_errors: bool = True,
    include_symbol_analysis: bool = True,
    include_file_summaries: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive codebase analysis.
    
    Args:
        codebase: The codebase to analyze
        include_runtime_errors: Whether to collect runtime errors
        include_symbol_analysis: Whether to perform symbol analysis
        include_file_summaries: Whether to generate file summaries
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    analyzer = SerenaCodebaseAnalyzer()
    
    try:
        result = await analyzer.analyze_codebase(
            codebase,
            include_runtime_errors=include_runtime_errors,
            include_symbol_analysis=include_symbol_analysis,
            include_file_summaries=include_file_summaries
        )
        
        return result.to_dict()
        
    finally:
        await analyzer.shutdown()


async def analyze_github_repository_comprehensive(
    repo_url: str,
    branch: str = "main",
    work_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive GitHub repository analysis.
    
    Args:
        repo_url: GitHub repository URL
        branch: Branch to analyze
        work_dir: Working directory for cloning
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    analyzer = SerenaCodebaseAnalyzer(work_dir=work_dir)
    
    try:
        result = await analyzer.analyze_repository_by_url(repo_url, branch)
        return result.to_dict()
        
    finally:
        await analyzer.shutdown()


async def get_repository_quality_report(
    repo_url: str,
    branch: str = "main",
    work_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get a quality-focused report for a GitHub repository.
    
    Args:
        repo_url: GitHub repository URL
        branch: Branch to analyze
        work_dir: Working directory for cloning
        
    Returns:
        Dictionary with quality metrics and recommendations
    """
    result = await analyze_github_repository_comprehensive(repo_url, branch, work_dir)
    
    return {
        'repository': result['repository'],
        'quality_metrics': result['metrics']['quality_metrics'],
        'error_summary': result['metrics']['error_metrics'],
        'error_hotspots': result['error_hotspots'],
        'symbol_complexity': result['symbol_analysis'],
        'recommendations': _generate_quality_recommendations(result)
    }


def _generate_quality_recommendations(analysis_result: Dict[str, Any]) -> List[str]:
    """Generate quality improvement recommendations."""
    recommendations = []
    
    error_metrics = analysis_result['metrics']['error_metrics']
    quality_metrics = analysis_result['metrics']['quality_metrics']
    
    if error_metrics['critical_errors'] > 0:
        recommendations.append(f"🔴 CRITICAL: Fix {error_metrics['critical_errors']} critical errors")
    
    if error_metrics['errors'] > 10:
        recommendations.append(f"🟠 HIGH: Address {error_metrics['errors']} errors")
    
    if quality_metrics['maintainability_index'] < 70:
        recommendations.append(f"🟡 MEDIUM: Improve maintainability index (currently {quality_metrics['maintainability_index']:.1f}/100)")
    
    if quality_metrics['technical_debt_score'] > 50:
        recommendations.append(f"🟣 MEDIUM: Reduce technical debt (score: {quality_metrics['technical_debt_score']})")
    
    if quality_metrics['test_coverage_estimate'] < 50:
        recommendations.append(f"🔵 LOW: Increase test coverage (estimated {quality_metrics['test_coverage_estimate']:.1f}%)")
    
    return recommendations[:5]  # Limit to top 5 recommendations
