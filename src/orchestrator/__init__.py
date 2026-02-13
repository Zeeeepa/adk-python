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

"""ADK Orchestrator Framework.

This module provides a comprehensive orchestration framework for Google ADK,
enabling sophisticated multi-agent workflows, hierarchical delegation,
and distributed agent communication.
"""

from __future__ import annotations

from .master_orchestrator import MasterOrchestrator
from .base_orchestrator import BaseOrchestrator
from .agent_registry import AgentRegistry
from .delegation_engine import DelegationEngine
from .workflow_engine import WorkflowEngine
from .state_manager import StateManager

__all__ = [
    'MasterOrchestrator',
    'BaseOrchestrator', 
    'AgentRegistry',
    'DelegationEngine',
    'WorkflowEngine',
    'StateManager',
]

__version__ = '1.0.0'

