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

"""Base environment class for execution environment abstraction."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional


class BaseEnvironment(ABC):
  """Abstract base class for execution environments.
  
  This class defines the interface that all execution environments must
  implement to provide consistent command execution and file operations
  across different contexts (local, WSL2, SSH, etc.).
  """
  
  def __init__(self, name: str):
    """Initialize the base environment.
    
    Args:
      name: Name of the environment
    """
    self.name = name
  
  @abstractmethod
  def is_available(self) -> bool:
    """Check if the environment is available and accessible.
    
    Returns:
      True if environment is available, False otherwise
    """
    pass
  
  @abstractmethod
  async def execute_command(
      self,
      command: str,
      timeout: Optional[float] = None,
      working_dir: Optional[str] = None
  ) -> Dict[str, Any]:
    """Execute a command in the environment.
    
    Args:
      command: Command to execute
      timeout: Command timeout in seconds
      working_dir: Working directory for the command
    
    Returns:
      Dictionary with execution results containing:
        - success: bool
        - stdout: str
        - stderr: str
        - return_code: int
        - execution_time: float
    """
    pass
  
  @abstractmethod
  async def read_file(self, file_path: str) -> str:
    """Read a file from the environment.
    
    Args:
      file_path: Path to the file to read
    
    Returns:
      File contents as string
    
    Raises:
      FileNotFoundError: If file doesn't exist
      PermissionError: If file can't be read
    """
    pass
  
  @abstractmethod
  async def write_file(self, file_path: str, content: str) -> bool:
    """Write content to a file in the environment.
    
    Args:
      file_path: Path to the file to write
      content: Content to write
    
    Returns:
      True if successful, False otherwise
    """
    pass
  
  @abstractmethod
  async def file_exists(self, file_path: str) -> bool:
    """Check if a file exists in the environment.
    
    Args:
      file_path: Path to check
    
    Returns:
      True if file exists, False otherwise
    """
    pass
  
  async def create_directory(self, dir_path: str) -> bool:
    """Create a directory in the environment.
    
    Args:
      dir_path: Path of directory to create
    
    Returns:
      True if successful, False otherwise
    """
    # Default implementation using command execution
    try:
      result = await self.execute_command(f"mkdir -p '{dir_path}'")
      return result.get('success', False)
    except Exception:
      return False
  
  async def list_directory(self, dir_path: str) -> Optional[list[str]]:
    """List contents of a directory.
    
    Args:
      dir_path: Path of directory to list
    
    Returns:
      List of file/directory names, or None if failed
    """
    # Default implementation using command execution
    try:
      result = await self.execute_command(f"ls -1 '{dir_path}'")
      if result.get('success', False):
        return result.get('stdout', '').strip().split('\n')
      return None
    except Exception:
      return None
  
  async def remove_file(self, file_path: str) -> bool:
    """Remove a file from the environment.
    
    Args:
      file_path: Path to the file to remove
    
    Returns:
      True if successful, False otherwise
    """
    # Default implementation using command execution
    try:
      result = await self.execute_command(f"rm -f '{file_path}'")
      return result.get('success', False)
    except Exception:
      return False
  
  async def copy_file(self, source: str, destination: str) -> bool:
    """Copy a file within the environment.
    
    Args:
      source: Source file path
      destination: Destination file path
    
    Returns:
      True if successful, False otherwise
    """
    # Default implementation using command execution
    try:
      result = await self.execute_command(f"cp '{source}' '{destination}'")
      return result.get('success', False)
    except Exception:
      return False
  
  async def get_working_directory(self) -> Optional[str]:
    """Get the current working directory.
    
    Returns:
      Current working directory path, or None if failed
    """
    # Default implementation using command execution
    try:
      result = await self.execute_command("pwd")
      if result.get('success', False):
        return result.get('stdout', '').strip()
      return None
    except Exception:
      return None
  
  async def set_working_directory(self, dir_path: str) -> bool:
    """Set the working directory.
    
    Args:
      dir_path: Directory path to set as working directory
    
    Returns:
      True if successful, False otherwise
    """
    # Default implementation using command execution
    try:
      result = await self.execute_command(f"cd '{dir_path}' && pwd")
      return result.get('success', False)
    except Exception:
      return False
  
  def get_info(self) -> Dict[str, Any]:
    """Get information about the environment.
    
    Returns:
      Dictionary with environment information
    """
    return {
        'name': self.name,
        'type': self.__class__.__name__,
        'available': self.is_available()
    }
  
  def detect_capabilities(self) -> Dict[str, bool]:
    """Detect capabilities of the environment.
    
    Returns:
      Dictionary of capability flags
    """
    # Base implementation - subclasses should override
    return {
        'command_execution': True,
        'file_operations': True,
    }
  
  def __str__(self) -> str:
    """String representation of the environment."""
    return f"{self.__class__.__name__}(name='{self.name}')"
  
  def __repr__(self) -> str:
    """Detailed string representation of the environment."""
    return f"{self.__class__.__name__}(name='{self.name}', available={self.is_available()})"

