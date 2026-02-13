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

"""Template validator for validating agent templates."""

from __future__ import annotations

import re
from typing import Any
from typing import Dict
from typing import List

from pydantic import BaseModel

from ..models.agent_template import AgentTemplate


class ValidationResult(BaseModel):
  """Result of template validation."""
  
  is_valid: bool
  """Whether the template is valid."""
  
  errors: List[str] = []
  """List of validation errors."""
  
  warnings: List[str] = []
  """List of validation warnings."""


class TemplateValidator:
  """Validator for agent templates.
  
  This class provides comprehensive validation of agent templates including
  schema validation, configuration validation, and best practice checks.
  """
  
  def __init__(self):
    """Initialize the template validator."""
    self.supported_agent_types = {
        "LlmAgent",
        "SequentialAgent", 
        "ParallelAgent",
        "LoopAgent"
    }
    
    self.required_llm_fields = {
        "model", "name"
    }
    
    self.supported_environments = {
        "local", "wsl2", "ssh"
    }
  
  def validate_template(self, template: AgentTemplate) -> ValidationResult:
    """Validate an agent template.
    
    Args:
      template: Template to validate
    
    Returns:
      Validation result with errors and warnings
    """
    result = ValidationResult(is_valid=True)
    
    # Basic field validation
    self._validate_basic_fields(template, result)
    
    # Agent type validation
    self._validate_agent_type(template, result)
    
    # Configuration validation
    self._validate_agent_config(template, result)
    
    # Environment validation
    self._validate_environments(template, result)
    
    # Dependency validation
    self._validate_dependencies(template, result)
    
    # Version validation
    self._validate_version(template, result)
    
    # Best practice checks
    self._check_best_practices(template, result)
    
    # Set overall validity
    result.is_valid = len(result.errors) == 0
    
    return result
  
  def _validate_basic_fields(self, template: AgentTemplate, result: ValidationResult) -> None:
    """Validate basic template fields.
    
    Args:
      template: Template to validate
      result: Validation result to update
    """
    # Name validation
    if not template.name:
      result.errors.append("Template name is required")
    elif not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', template.name):
      result.errors.append("Template name must start with a letter and contain only letters, numbers, underscores, and hyphens")
    
    # Description validation
    if not template.description:
      result.warnings.append("Template description is empty - consider adding a description")
    elif len(template.description) > 500:
      result.warnings.append("Template description is very long - consider keeping it concise")
    
    # ID validation
    if not template.id:
      result.errors.append("Template ID is required")
  
  def _validate_agent_type(self, template: AgentTemplate, result: ValidationResult) -> None:
    """Validate agent type.
    
    Args:
      template: Template to validate
      result: Validation result to update
    """
    if not template.agent_type:
      result.errors.append("Agent type is required")
    elif template.agent_type not in self.supported_agent_types:
      result.errors.append(f"Unsupported agent type: {template.agent_type}. Supported types: {', '.join(self.supported_agent_types)}")
  
  def _validate_agent_config(self, template: AgentTemplate, result: ValidationResult) -> None:
    """Validate agent configuration.
    
    Args:
      template: Template to validate
      result: Validation result to update
    """
    if not template.agent_config:
      result.errors.append("Agent configuration is required")
      return
    
    config = template.agent_config
    
    # Validate based on agent type
    if template.agent_type == "LlmAgent":
      self._validate_llm_agent_config(config, result)
    elif template.agent_type in ["SequentialAgent", "ParallelAgent", "LoopAgent"]:
      self._validate_workflow_agent_config(config, result)
  
  def _validate_llm_agent_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
    """Validate LLM agent configuration.
    
    Args:
      config: Agent configuration
      result: Validation result to update
    """
    # Check required fields
    for field in self.required_llm_fields:
      if field not in config:
        result.errors.append(f"LlmAgent configuration missing required field: {field}")
    
    # Validate model
    if "model" in config:
      model = config["model"]
      if not isinstance(model, str) or not model:
        result.errors.append("Model must be a non-empty string")
    
    # Validate instruction
    if "instruction" in config:
      instruction = config["instruction"]
      if instruction and len(instruction) > 10000:
        result.warnings.append("Instruction is very long - consider breaking it down")
    
    # Validate tools
    if "tools" in config:
      tools = config["tools"]
      if not isinstance(tools, list):
        result.errors.append("Tools must be a list")
      elif len(tools) > 50:
        result.warnings.append("Large number of tools may impact performance")
  
  def _validate_workflow_agent_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
    """Validate workflow agent configuration.
    
    Args:
      config: Agent configuration
      result: Validation result to update
    """
    # Check for sub_agents
    if "sub_agents" not in config:
      result.warnings.append("Workflow agent has no sub-agents defined")
    else:
      sub_agents = config["sub_agents"]
      if not isinstance(sub_agents, list):
        result.errors.append("sub_agents must be a list")
      elif len(sub_agents) == 0:
        result.warnings.append("Workflow agent has empty sub_agents list")
      elif len(sub_agents) > 20:
        result.warnings.append("Large number of sub-agents may impact performance")
    
    # Validate max_iterations for LoopAgent
    if "max_iterations" in config:
      max_iterations = config["max_iterations"]
      if not isinstance(max_iterations, int) or max_iterations <= 0:
        result.errors.append("max_iterations must be a positive integer")
      elif max_iterations > 1000:
        result.warnings.append("Very high max_iterations may cause long execution times")
  
  def _validate_environments(self, template: AgentTemplate, result: ValidationResult) -> None:
    """Validate required environments.
    
    Args:
      template: Template to validate
      result: Validation result to update
    """
    for env in template.required_environments:
      if env not in self.supported_environments:
        result.errors.append(f"Unsupported environment: {env}. Supported environments: {', '.join(self.supported_environments)}")
  
  def _validate_dependencies(self, template: AgentTemplate, result: ValidationResult) -> None:
    """Validate template dependencies.
    
    Args:
      template: Template to validate
      result: Validation result to update
    """
    # Check for circular dependencies (basic check)
    if template.parent_template_id and template.parent_template_id == template.id:
      result.errors.append("Template cannot be its own parent")
    
    # Check dependency count
    if len(template.dependencies) > 10:
      result.warnings.append("Large number of dependencies may complicate template management")
  
  def _validate_version(self, template: AgentTemplate, result: ValidationResult) -> None:
    """Validate template version.
    
    Args:
      template: Template to validate
      result: Validation result to update
    """
    version = template.version
    
    # Basic semantic version check
    if not re.match(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$', version):
      result.warnings.append("Version should follow semantic versioning (e.g., 1.0.0)")
  
  def _check_best_practices(self, template: AgentTemplate, result: ValidationResult) -> None:
    """Check best practices.
    
    Args:
      template: Template to validate
      result: Validation result to update
    """
    # Check for tags
    if not template.tags:
      result.warnings.append("Consider adding tags to improve template discoverability")
    
    # Check for author
    if not template.author:
      result.warnings.append("Consider specifying an author for the template")
    
    # Check description length
    if template.description and len(template.description) < 20:
      result.warnings.append("Consider providing a more detailed description")
    
    # Check for ADK version constraints
    if not template.min_adk_version:
      result.warnings.append("Consider specifying minimum ADK version for compatibility")
    
    # Check configuration complexity
    if template.agent_config and len(str(template.agent_config)) > 5000:
      result.warnings.append("Complex configuration may be difficult to maintain")
  
  def validate_template_compatibility(
      self,
      template: AgentTemplate,
      target_environment: str,
      adk_version: str
  ) -> ValidationResult:
    """Validate template compatibility with environment and ADK version.
    
    Args:
      template: Template to validate
      target_environment: Target environment
      adk_version: ADK version
    
    Returns:
      Validation result for compatibility
    """
    result = ValidationResult(is_valid=True)
    
    # Environment compatibility
    if template.required_environments and target_environment not in template.required_environments:
      result.errors.append(f"Template requires environments {template.required_environments} but target is {target_environment}")
    
    # ADK version compatibility (simplified check)
    if template.min_adk_version and adk_version < template.min_adk_version:
      result.errors.append(f"Template requires ADK version >= {template.min_adk_version} but current is {adk_version}")
    
    if template.max_adk_version and adk_version > template.max_adk_version:
      result.errors.append(f"Template requires ADK version <= {template.max_adk_version} but current is {adk_version}")
    
    result.is_valid = len(result.errors) == 0
    return result

