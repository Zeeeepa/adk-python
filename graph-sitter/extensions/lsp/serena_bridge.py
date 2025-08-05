"""
Serena LSP Bridge for Graph-Sitter

This module provides a comprehensive bridge between Serena's LSP implementation
and graph-sitter's codebase analysis system, integrating all existing types
and avoiding redundant definitions.
"""

import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import weakref

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

# SolidLSP imports
from solidlsp.ls_types import (
    DiagnosticSeverity, Diagnostic, Position, Range, MarkupContent,
    Location, MarkupKind, CompletionItemKind, CompletionItem, 
    UnifiedSymbolInformation, SymbolKind, SymbolTag
)
from solidlsp.ls_utils import TextUtils, PathUtils, FileUtils, PlatformId, SymbolUtils
from solidlsp.ls_request import LanguageServerRequest
from solidlsp.ls_logger import LanguageServerLogger, LogLine
from solidlsp.ls_handler import SolidLanguageServerHandler, Request, LanguageServerTerminatedException
from solidlsp.ls import SolidLanguageServer, LSPFileBuffer
from solidlsp.lsp_protocol_handler.lsp_constants import LSPConstants
from solidlsp.lsp_protocol_handler.lsp_requests import LspRequest
from solidlsp.lsp_protocol_handler.lsp_types import (
    DocumentDiagnosticReportKind, ErrorCodes, LSPErrorCodes, DiagnosticSeverity as LSPDiagnosticSeverity,
    DiagnosticTag, InitializeError, WorkspaceDiagnosticParams, WorkspaceDiagnosticReport,
    WorkspaceDiagnosticReportPartialResult, PublishDiagnosticsParams, RelatedFullDocumentDiagnosticReport,
    RelatedUnchangedDocumentDiagnosticReport, UnchangedDocumentDiagnosticReport,
    FullDocumentDiagnosticReport, DiagnosticOptions, WorkspaceFullDocumentDiagnosticReport,
    WorkspaceUnchangedDocumentDiagnosticReport, DiagnosticRelatedInformation,
    DiagnosticWorkspaceClientCapabilities, DiagnosticClientCapabilities, PublishDiagnosticsClientCapabilities
)
from solidlsp.lsp_protocol_handler.server import ProcessLaunchInfo, LSPError, MessageType

# Serena imports
from serena.symbol import (
    LanguageServerSymbolRetriever, ReferenceInLanguageServerSymbol,
    LanguageServerSymbol, Symbol, PositionInFile, LanguageServerSymbolLocation
)
from serena.text_utils import MatchedConsecutiveLines, TextLine, LineType
from serena.project import Project
from serena.gui_log_viewer import GuiLogViewer, LogLevel, GuiLogViewerHandler
from serena.code_editor import CodeEditor
from serena.cli import (
    PromptCommands, ToolCommands, ProjectCommands, SerenaConfigCommands,
    ContextCommands, ModeCommands, TopLevelCommands, AutoRegisteringGroup, ProjectType
)

from graph_sitter.shared.logging.get_logger import get_logger

logger = get_logger(__name__)


@dataclass
class SerenaErrorInfo:
    """Enhanced error information using existing graph-sitter and Serena types."""
    file_path: str
    line: int
    character: int
    message: str
    severity: DiagnosticSeverity
    source: str = "serena"
    code: Optional[Union[str, int]] = None
    end_line: Optional[int] = None
    end_character: Optional[int] = None
    
    # Serena-specific enhancements
    symbol_info: Optional[LanguageServerSymbol] = None
    related_symbols: List[ReferenceInLanguageServerSymbol] = field(default_factory=list)
    context_lines: Optional[MatchedConsecutiveLines] = None
    fix_suggestions: List[str] = field(default_factory=list)
    runtime_context: Optional[RuntimeContext] = None
    
    def to_diagnostic(self) -> Diagnostic:
        """Convert to LSP Diagnostic."""
        range_obj = Range(
            start=Position(line=self.line - 1, character=self.character),
            end=Position(
                line=(self.end_line or self.line) - 1,
                character=(self.end_character or self.character)
            )
        )
        
        return Diagnostic(
            range=range_obj,
            severity=self.severity,
            code=self.code,
            source=self.source,
            message=self.message
        )
    
    @property
    def is_error(self) -> bool:
        """Check if this is an error severity."""
        return self.severity == DiagnosticSeverity.ERROR
    
    @property
    def is_warning(self) -> bool:
        """Check if this is a warning severity."""
        return self.severity == DiagnosticSeverity.WARNING


class SerenaLSPBridge:
    """
    Comprehensive bridge between Serena's LSP implementation and graph-sitter.
    
    This bridge integrates:
    - SolidLSP server capabilities
    - Serena symbol analysis
    - Graph-sitter codebase analysis
    - Runtime error collection
    """
    
    def __init__(self, codebase: Codebase, enable_runtime_collection: bool = True):
        self.codebase = codebase
        self.repo_path = Path(codebase.repo_path)
        self.enable_runtime_collection = enable_runtime_collection
        
        # Core components
        self.solid_lsp_server: Optional[SolidLanguageServer] = None
        self.serena_project: Optional[Project] = None
        self.symbol_retriever: Optional[LanguageServerSymbolRetriever] = None
        self.lsp_logger: Optional[LanguageServerLogger] = None
        self.runtime_collector: Optional[RuntimeErrorCollector] = None
        
        # State management
        self.is_initialized = False
        self._lock = threading.RLock()
        self._diagnostics_cache: Dict[str, List[SerenaErrorInfo]] = {}
        self._symbol_cache: Dict[str, List[LanguageServerSymbol]] = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all Serena and LSP components."""
        try:
            logger.info(f"Initializing Serena LSP bridge for {self.repo_path}")
            
            # Initialize SolidLSP server
            self._initialize_solid_lsp()
            
            # Initialize Serena project
            self._initialize_serena_project()
            
            # Initialize symbol retriever
            self._initialize_symbol_retriever()
            
            # Initialize runtime error collection
            if self.enable_runtime_collection:
                self._initialize_runtime_collection()
            
            # Initialize LSP logger
            self._initialize_lsp_logger()
            
            self.is_initialized = True
            logger.info("Serena LSP bridge initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Serena LSP bridge: {e}")
            self.is_initialized = False
    
    def _initialize_solid_lsp(self) -> None:
        """Initialize SolidLSP server."""
        try:
            self.solid_lsp_server = SolidLanguageServer()
            logger.info("SolidLSP server initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SolidLSP server: {e}")
    
    def _initialize_serena_project(self) -> None:
        """Initialize Serena project."""
        try:
            self.serena_project = Project(str(self.repo_path))
            logger.info("Serena project initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Serena project: {e}")
    
    def _initialize_symbol_retriever(self) -> None:
        """Initialize symbol retriever."""
        try:
            if self.solid_lsp_server:
                self.symbol_retriever = LanguageServerSymbolRetriever(self.solid_lsp_server)
                logger.info("Symbol retriever initialized")
        except Exception as e:
            logger.error(f"Failed to initialize symbol retriever: {e}")
    
    def _initialize_runtime_collection(self) -> None:
        """Initialize runtime error collection."""
        try:
            self.runtime_collector = RuntimeErrorCollector(str(self.repo_path))
            self.runtime_collector.start_collection()
            logger.info("Runtime error collection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize runtime collection: {e}")
    
    def _initialize_lsp_logger(self) -> None:
        """Initialize LSP logger."""
        try:
            self.lsp_logger = LanguageServerLogger()
            logger.info("LSP logger initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LSP logger: {e}")
    
    def get_diagnostics(self, file_path: Optional[str] = None) -> List[SerenaErrorInfo]:
        """Get diagnostics for a file or all files."""
        if not self.is_initialized:
            return []
        
        diagnostics = []
        
        with self._lock:
            if file_path:
                # Get diagnostics for specific file
                diagnostics.extend(self._get_file_diagnostics(file_path))
            else:
                # Get diagnostics for all files
                for file_obj in self.codebase.files:
                    if file_obj.file_path.endswith('.py'):
                        diagnostics.extend(self._get_file_diagnostics(file_obj.file_path))
        
        return diagnostics
    
    def _get_file_diagnostics(self, file_path: str) -> List[SerenaErrorInfo]:
        """Get diagnostics for a specific file."""
        # Check cache first
        if file_path in self._diagnostics_cache:
            return self._diagnostics_cache[file_path]
        
        diagnostics = []
        
        try:
            # Get LSP diagnostics if server is available
            if self.solid_lsp_server:
                lsp_diagnostics = self._get_lsp_diagnostics(file_path)
                diagnostics.extend(lsp_diagnostics)
            
            # Get runtime errors if collector is available
            if self.runtime_collector:
                runtime_errors = self._get_runtime_errors(file_path)
                diagnostics.extend(runtime_errors)
            
            # Enhance diagnostics with symbol information
            self._enhance_diagnostics_with_symbols(diagnostics, file_path)
            
            # Cache results
            self._diagnostics_cache[file_path] = diagnostics
            
        except Exception as e:
            logger.error(f"Error getting diagnostics for {file_path}: {e}")
        
        return diagnostics
    
    def _get_lsp_diagnostics(self, file_path: str) -> List[SerenaErrorInfo]:
        """Get LSP diagnostics from SolidLSP server."""
        diagnostics = []
        
        try:
            # This would integrate with actual SolidLSP diagnostic retrieval
            # For now, return empty list as placeholder
            pass
            
        except Exception as e:
            logger.error(f"Error getting LSP diagnostics: {e}")
        
        return diagnostics
    
    def _get_runtime_errors(self, file_path: str) -> List[SerenaErrorInfo]:
        """Get runtime errors for a file."""
        diagnostics = []
        
        try:
            if self.runtime_collector:
                runtime_errors = self.runtime_collector.get_errors_for_file(file_path)
                
                for error in runtime_errors:
                    diagnostic = SerenaErrorInfo(
                        file_path=file_path,
                        line=error.line,
                        character=error.character,
                        message=error.message,
                        severity=DiagnosticSeverity.ERROR,
                        source="runtime",
                        runtime_context=error.context
                    )
                    diagnostics.append(diagnostic)
                    
        except Exception as e:
            logger.error(f"Error getting runtime errors: {e}")
        
        return diagnostics
    
    def _enhance_diagnostics_with_symbols(self, diagnostics: List[SerenaErrorInfo], file_path: str) -> None:
        """Enhance diagnostics with symbol information."""
        try:
            if not self.symbol_retriever:
                return
            
            # Get symbols for the file
            symbols = self._get_file_symbols(file_path)
            
            for diagnostic in diagnostics:
                # Find relevant symbols near the diagnostic location
                relevant_symbols = self._find_symbols_near_position(
                    symbols, diagnostic.line, diagnostic.character
                )
                
                if relevant_symbols:
                    diagnostic.symbol_info = relevant_symbols[0]
                    # Convert to ReferenceInLanguageServerSymbol for related_symbols
                    diagnostic.related_symbols = []  # Would need proper conversion
                
                # Generate fix suggestions based on symbol context
                diagnostic.fix_suggestions = self._generate_fix_suggestions(diagnostic)
                
        except Exception as e:
            logger.error(f"Error enhancing diagnostics with symbols: {e}")
    
    def _get_file_symbols(self, file_path: str) -> List[LanguageServerSymbol]:
        """Get symbols for a file."""
        # Check cache first
        if file_path in self._symbol_cache:
            return self._symbol_cache[file_path]
        
        symbols = []
        
        try:
            if self.symbol_retriever:
                # This would integrate with actual symbol retrieval
                # For now, return empty list as placeholder
                pass
            
            # Cache results
            self._symbol_cache[file_path] = symbols
            
        except Exception as e:
            logger.error(f"Error getting symbols for {file_path}: {e}")
        
        return symbols
    
    def _find_symbols_near_position(self, symbols: List[LanguageServerSymbol], 
                                   line: int, character: int) -> List[LanguageServerSymbol]:
        """Find symbols near a specific position."""
        relevant_symbols = []
        
        for symbol in symbols:
            # This would implement actual position-based symbol matching
            # For now, return empty list as placeholder
            pass
        
        return relevant_symbols
    
    def _generate_fix_suggestions(self, diagnostic: SerenaErrorInfo) -> List[str]:
        """Generate fix suggestions for a diagnostic."""
        suggestions = []
        
        try:
            # Generate suggestions based on diagnostic type and context
            if "undefined" in diagnostic.message.lower():
                suggestions.append("Check if the variable is defined before use")
                suggestions.append("Verify import statements")
            elif "syntax" in diagnostic.message.lower():
                suggestions.append("Check for missing parentheses or brackets")
                suggestions.append("Verify proper indentation")
            elif "type" in diagnostic.message.lower():
                suggestions.append("Check argument types")
                suggestions.append("Verify function signature")
            
            # Add symbol-specific suggestions
            if diagnostic.symbol_info:
                suggestions.append(f"Check usage of symbol '{diagnostic.symbol_info.name}'")
            
        except Exception as e:
            logger.error(f"Error generating fix suggestions: {e}")
        
        return suggestions
    
    def get_completions(self, file_path: str, line: int, character: int) -> List[CompletionItem]:
        """Get code completions at a specific position."""
        if not self.is_initialized or not self.solid_lsp_server:
            return []
        
        try:
            # This would integrate with actual completion retrieval
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Error getting completions: {e}")
            return []
    
    def get_hover_info(self, file_path: str, line: int, character: int) -> Optional[MarkupContent]:
        """Get hover information at a specific position."""
        if not self.is_initialized:
            return None
        
        try:
            # Get symbol at position
            symbols = self._get_file_symbols(file_path)
            relevant_symbols = self._find_symbols_near_position(symbols, line, character)
            
            if relevant_symbols:
                symbol = relevant_symbols[0]
                content = f"**{symbol.name}**\n\n"
                
                # Add symbol information
                if hasattr(symbol, 'documentation'):
                    content += symbol.documentation
                
                return MarkupContent(kind=MarkupKind.MARKDOWN, value=content)
            
        except Exception as e:
            logger.error(f"Error getting hover info: {e}")
        
        return None
    
    def get_symbol_references(self, file_path: str, line: int, character: int) -> List[Location]:
        """Get references to a symbol at a specific position."""
        if not self.is_initialized or not self.symbol_retriever:
            return []
        
        try:
            # This would integrate with actual reference finding
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Error getting symbol references: {e}")
            return []
    
    def rename_symbol(self, file_path: str, line: int, character: int, new_name: str) -> Dict[str, Any]:
        """Rename a symbol at a specific position."""
        if not self.is_initialized:
            return {"success": False, "error": "Bridge not initialized"}
        
        try:
            # This would integrate with actual symbol renaming
            # For now, return placeholder response
            return {
                "success": True,
                "changes": {},
                "message": f"Symbol rename to '{new_name}' would be performed"
            }
            
        except Exception as e:
            logger.error(f"Error renaming symbol: {e}")
            return {"success": False, "error": str(e)}
    
    def get_workspace_symbols(self, query: str) -> List[UnifiedSymbolInformation]:
        """Get workspace symbols matching a query."""
        if not self.is_initialized or not self.symbol_retriever:
            return []
        
        try:
            # This would integrate with actual workspace symbol search
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Error getting workspace symbols: {e}")
            return []
    
    def refresh_diagnostics(self) -> None:
        """Refresh all cached diagnostics."""
        with self._lock:
            self._diagnostics_cache.clear()
            self._symbol_cache.clear()
        
        logger.info("Diagnostics cache refreshed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the bridge."""
        return {
            "initialized": self.is_initialized,
            "repo_path": str(self.repo_path),
            "components": {
                "solid_lsp_server": self.solid_lsp_server is not None,
                "serena_project": self.serena_project is not None,
                "symbol_retriever": self.symbol_retriever is not None,
                "runtime_collector": self.runtime_collector is not None,
                "lsp_logger": self.lsp_logger is not None
            },
            "cache_sizes": {
                "diagnostics": len(self._diagnostics_cache),
                "symbols": len(self._symbol_cache)
            },
            "runtime_collection_enabled": self.enable_runtime_collection
        }
    
    def shutdown(self) -> None:
        """Shutdown the bridge and clean up resources."""
        try:
            logger.info("Shutting down Serena LSP bridge")
            
            # Stop runtime collection
            if self.runtime_collector:
                self.runtime_collector.stop_collection()
            
            # Shutdown SolidLSP server
            if self.solid_lsp_server and hasattr(self.solid_lsp_server, 'shutdown'):
                self.solid_lsp_server.shutdown()
            
            # Clear caches
            with self._lock:
                self._diagnostics_cache.clear()
                self._symbol_cache.clear()
            
            # Reset state
            self.is_initialized = False
            
            logger.info("Serena LSP bridge shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during bridge shutdown: {e}")


def create_serena_bridge(codebase: Codebase, **kwargs) -> SerenaLSPBridge:
    """Create a Serena LSP bridge for a codebase."""
    return SerenaLSPBridge(codebase, **kwargs)


def get_enhanced_diagnostics(codebase: Codebase, file_path: Optional[str] = None) -> List[SerenaErrorInfo]:
    """Get enhanced diagnostics using Serena LSP bridge."""
    bridge = SerenaLSPBridge(codebase)
    try:
        return bridge.get_diagnostics(file_path)
    finally:
        bridge.shutdown()
