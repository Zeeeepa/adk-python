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

"""Environment manager for handling different execution environments."""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .environments.base_environment import BaseEnvironment
from .environments.local_environment import LocalEnvironment
from .environments.ssh_environment import SSHEnvironment
from .environments.wsl_environment import WSLEnvironment

logger = logging.getLogger('google_adk.orchestrator.environment_manager')


class EnvironmentManager:
  """Manager for different execution environments.
  
  The EnvironmentManager provides abstraction over different execution
  environments including local, WSL2, and SSH environments. It handles
  environment detection, switching, and provides consistent APIs for
  command execution and file operations.
  """
  
  def __init__(self):
    """Initialize the environment manager."""
    self._environments: Dict[str, BaseEnvironment] = {}
    self._active_environment: Optional[str] = None
    
    # Initialize available environments
    self._initialize_environments()
    
    # Set default environment
    self._set_default_environment()
    
    logger.info(f"Environment manager initialized with {len(self._environments)} environments")
  
  def _initialize_environments(self) -> None:
    """Initialize available environments."""
    # Always available: local environment
    self._environments['local'] = LocalEnvironment()
    
    # Check for WSL2 availability
    if self._is_wsl_available():
      self._environments['wsl2'] = WSLEnvironment()
      logger.info("WSL2 environment detected and initialized")
    
    # SSH environments will be added dynamically
    logger.info(f"Initialized {len(self._environments)} base environments")
  
  def _is_wsl_available(self) -> bool:
    """Check if WSL2 is available on the system."""
    try:
      # Check if we're running on Windows
      if platform.system() != 'Windows':
        return False
      
      # Check if WSL is installed and has distributions
      result = subprocess.run(
          ['wsl', '--list', '--quiet'],
          capture_output=True,
          text=True,
          timeout=5
      )
      
      return result.returncode == 0 and result.stdout.strip()
      
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
      return False
  
  def _set_default_environment(self) -> None:
    """Set the default active environment."""
    # Prefer WSL2 if available, otherwise use local
    if 'wsl2' in self._environments:
      self._active_environment = 'wsl2'
    else:
      self._active_environment = 'local'
    
    logger.info(f"Default environment set to: {self._active_environment}")
  
  def get_available_environments(self) -> List[str]:
    """Get list of available environment names.
    
    Returns:
      List of available environment names
    """
    return list(self._environments.keys())
  
  def get_active_environment(self) -> str:
    """Get the currently active environment name.
    
    Returns:
      Name of the active environment
    """
    return self._active_environment or 'local'
  
  def set_active_environment(self, environment: str) -> bool:
    """Set the active environment.
    
    Args:
      environment: Name of the environment to activate
    
    Returns:
      True if environment was set successfully, False otherwise
    """
    if environment not in self._environments:
      logger.error(f"Environment '{environment}' not available")
      return False
    
    # Test environment connectivity
    env = self._environments[environment]
    if not env.is_available():
      logger.error(f"Environment '{environment}' is not available")
      return False
    
    self._active_environment = environment
    logger.info(f"Active environment set to: {environment}")
    return True
  
  def get_environment(self, environment: Optional[str] = None) -> BaseEnvironment:
    """Get an environment instance.
    
    Args:
      environment: Environment name (uses active if not specified)
    
    Returns:
      Environment instance
    
    Raises:
      ValueError: If environment is not available
    """
    env_name = environment or self.get_active_environment()
    
    if env_name not in self._environments:
      raise ValueError(f"Environment '{env_name}' not available")
    
    return self._environments[env_name]
  
  def add_ssh_environment(
      self,
      name: str,
      host: str,
      username: str,
      password: Optional[str] = None,
      key_file: Optional[str] = None,
      port: int = 22
  ) -> bool:
    """Add an SSH environment.
    
    Args:
      name: Name for the SSH environment
      host: SSH host address
      username: SSH username
      password: SSH password (if not using key)
      key_file: Path to SSH private key file
      port: SSH port (default 22)
    
    Returns:
      True if environment was added successfully, False otherwise
    """
    try:
      ssh_env = SSHEnvironment(
          host=host,
          username=username,
          password=password,
          key_file=key_file,
          port=port
      )
      
      # Test connectivity
      if not ssh_env.is_available():
        logger.error(f"Cannot connect to SSH environment '{name}'")
        return False
      
      self._environments[name] = ssh_env
      logger.info(f"Added SSH environment '{name}' ({host}:{port})")
      return True
      
    except Exception as e:
      logger.error(f"Failed to add SSH environment '{name}': {e}")
      return False
  
  def remove_environment(self, environment: str) -> bool:
    """Remove an environment.
    
    Args:
      environment: Name of the environment to remove
    
    Returns:
      True if environment was removed, False otherwise
    """
    if environment in ['local', 'wsl2']:
      logger.error(f"Cannot remove built-in environment '{environment}'")
      return False
    
    if environment not in self._environments:
      logger.warning(f"Environment '{environment}' not found")
      return False
    
    # Close environment if needed
    env = self._environments[environment]
    if hasattr(env, 'close'):
      env.close()
    
    del self._environments[environment]
    
    # Switch to default if this was active
    if self._active_environment == environment:
      self._set_default_environment()
    
    logger.info(f"Removed environment '{environment}'")
    return True
  
  async def execute_command(
      self,
      command: str,
      environment: Optional[str] = None,
      timeout: Optional[float] = None,
      working_dir: Optional[str] = None
  ) -> Dict[str, Any]:
    """Execute a command in the specified environment.
    
    Args:
      command: Command to execute
      environment: Environment to use (active if not specified)
      timeout: Command timeout in seconds
      working_dir: Working directory for the command
    
    Returns:
      Dictionary with execution results
    """
    env = self.get_environment(environment)
    
    try:
      result = await env.execute_command(
          command=command,
          timeout=timeout,
          working_dir=working_dir
      )
      
      logger.debug(f"Command executed in {env.name}: {command}")
      return result
      
    except Exception as e:
      logger.error(f"Command execution failed in {env.name}: {e}")
      return {
          'success': False,
          'error': str(e),
          'stdout': '',
          'stderr': str(e),
          'return_code': -1
      }
  
  async def read_file(
      self,
      file_path: str,
      environment: Optional[str] = None
  ) -> Optional[str]:
    """Read a file from the specified environment.
    
    Args:
      file_path: Path to the file
      environment: Environment to use (active if not specified)
    
    Returns:
      File contents or None if failed
    """
    env = self.get_environment(environment)
    
    try:
      content = await env.read_file(file_path)
      logger.debug(f"File read from {env.name}: {file_path}")
      return content
      
    except Exception as e:
      logger.error(f"Failed to read file from {env.name}: {e}")
      return None
  
  async def write_file(
      self,
      file_path: str,
      content: str,
      environment: Optional[str] = None
  ) -> bool:
    """Write a file to the specified environment.
    
    Args:
      file_path: Path to the file
      content: Content to write
      environment: Environment to use (active if not specified)
    
    Returns:
      True if successful, False otherwise
    """
    env = self.get_environment(environment)
    
    try:
      success = await env.write_file(file_path, content)
      if success:
        logger.debug(f"File written to {env.name}: {file_path}")
      else:
        logger.error(f"Failed to write file to {env.name}: {file_path}")
      return success
      
    except Exception as e:
      logger.error(f"Failed to write file to {env.name}: {e}")
      return False
  
  async def file_exists(
      self,
      file_path: str,
      environment: Optional[str] = None
  ) -> bool:
    """Check if a file exists in the specified environment.
    
    Args:
      file_path: Path to check
      environment: Environment to use (active if not specified)
    
    Returns:
      True if file exists, False otherwise
    """
    env = self.get_environment(environment)
    
    try:
      exists = await env.file_exists(file_path)
      return exists
      
    except Exception as e:
      logger.error(f"Failed to check file existence in {env.name}: {e}")
      return False
  
  def get_environment_info(self, environment: Optional[str] = None) -> Dict[str, Any]:
    """Get information about an environment.
    
    Args:
      environment: Environment name (active if not specified)
    
    Returns:
      Dictionary with environment information
    """
    env = self.get_environment(environment)
    
    info = {
        'name': env.name,
        'type': env.__class__.__name__,
        'available': env.is_available(),
        'active': environment == self.get_active_environment()
    }
    
    # Add environment-specific info
    if hasattr(env, 'get_info'):
      info.update(env.get_info())
    
    return info
  
  def get_all_environments_info(self) -> Dict[str, Dict[str, Any]]:
    """Get information about all environments.
    
    Returns:
      Dictionary mapping environment names to their info
    """
    return {
        name: self.get_environment_info(name)
        for name in self._environments.keys()
    }
  
  def detect_environment_capabilities(self, environment: Optional[str] = None) -> Dict[str, bool]:
    """Detect capabilities of an environment.
    
    Args:
      environment: Environment name (active if not specified)
    
    Returns:
      Dictionary of capability flags
    """
    env = self.get_environment(environment)
    
    capabilities = {
        'command_execution': True,  # All environments support this
        'file_operations': True,    # All environments support this
        'python_available': False,
        'docker_available': False,
        'git_available': False,
        'node_available': False,
    }
    
    # Test for specific tools (this would be implemented in each environment)
    if hasattr(env, 'detect_capabilities'):
      capabilities.update(env.detect_capabilities())
    
    return capabilities

