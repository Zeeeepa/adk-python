"""
LSP Extensions for Graph-Sitter

This package provides Language Server Protocol (LSP) integration and
comprehensive error analysis capabilities for graph-sitter.
"""

# Import Serena LSP bridge
from .serena_bridge import (
    SerenaErrorInfo,
    SerenaLSPBridge,
    create_serena_bridge,
    get_enhanced_diagnostics,
)

# Import Serena analysis
from .serena_analysis import (
    RepositoryInfo,
    AnalysisMetrics,
    ComprehensiveAnalysisResult,
    SerenaCodebaseAnalyzer,
    analyze_codebase_comprehensive,
    analyze_github_repository_comprehensive,
    get_repository_quality_report,
)

__all__ = [
    # Serena LSP Bridge
    "SerenaErrorInfo",
    "SerenaLSPBridge",
    "create_serena_bridge",
    "get_enhanced_diagnostics",
    
    # Serena Analysis
    "RepositoryInfo",
    "AnalysisMetrics",
    "ComprehensiveAnalysisResult",
    "SerenaCodebaseAnalyzer",
    "analyze_codebase_comprehensive",
    "analyze_github_repository_comprehensive",
    "get_repository_quality_report",
]
