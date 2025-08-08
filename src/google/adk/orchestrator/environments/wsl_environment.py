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

"""WSL2 environment implementation for Windows Subsystem for Linux execution."""

from __future__ import annotations

import asyncio
import platform
import subprocess
import time
from typing import Any
from typing import Dict
from typing import Optional

from typing_extensions import override

from .base_environment import BaseEnvironment


class WSLEnvironment(BaseEnvironment):
  """WSL2 execution environment.
  
  This environment executes commands and performs file operations within
  Windows Subsystem for Linux (WSL2).
  """
  
  def __init__(self, distribution: Optional[str] = None):
    """Initialize the WSL environment.
    
    Args:
      distribution: WSL distribution name (uses default if not specified)
    """
    super().__init__("wsl2")
    self.distribution = distribution
    self._working_dir: Optional[str] = None
  
  @override
  def is_available(self) -> bool:
    """Check if WSL2 is available.
    
    Returns:
      True if WSL2 is available, False otherwise
    """
    try:
      # Check if we're on Windows
      if platform.system() != 'Windows':
        return False
      
      # Check if WSL is installed and has distributions
      result = subprocess.run(
          ['wsl', '--list', '--quiet'],
          capture_output=True,
          text=True,
          timeout=5
      )
      
      if result.returncode != 0:
        return False
      
      distributions = result.stdout.strip().split('\n')
      distributions = [d.strip() for d in distributions if d.strip()]
      
      if not distributions:
        return False
      
      # If specific distribution is requested, check if it exists
      if self.distribution:
        return self.distribution in distributions
      
      return True
      
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
      return False
  
  def _build_wsl_command(self, command: str) -> list[str]:
    """Build WSL command with proper distribution handling.
    
    Args:
      command: Command to execute in WSL
    
    Returns:
      List of command parts for subprocess
    """
    wsl_cmd = ['wsl']
    
    if self.distribution:
      wsl_cmd.extend(['--distribution', self.distribution])
    
    wsl_cmd.extend(['--', 'bash', '-c', command])
    
    return wsl_cmd
  
  @override
  async def execute_command(
      self,
      command: str,
      timeout: Optional[float] = None,
      working_dir: Optional[str] = None
  ) -> Dict[str, Any]:
    """Execute a command in WSL2.
    
    Args:
      command: Command to execute
      timeout: Command timeout in seconds
      working_dir: Working directory for the command
    
    Returns:
      Dictionary with execution results
    """
    start_time = time.time()
    
    try:
      # Prepare command with working directory
      if working_dir or self._working_dir:
        cwd = working_dir or self._working_dir
        command = f"cd '{cwd}' && {command}"
      
      # Build WSL command
      wsl_cmd = self._build_wsl_command(command)
      
      # Create subprocess
      process = await asyncio.create_subprocess_exec(
          *wsl_cmd,
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE
      )
      
      # Wait for completion with timeout
      try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )
      except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Command timed out',
            'return_code': -1,
            'execution_time': time.time() - start_time
        }
      
      execution_time = time.time() - start_time
      
      return {
          'success': process.returncode == 0,
          'stdout': stdout.decode('utf-8', errors='replace'),
          'stderr': stderr.decode('utf-8', errors='replace'),
          'return_code': process.returncode,
          'execution_time': execution_time
      }
      
    except Exception as e:
      return {
          'success': False,
          'stdout': '',
          'stderr': str(e),
          'return_code': -1,
          'execution_time': time.time() - start_time
      }
  
  @override
  async def read_file(self, file_path: str) -> str:
    """Read a file from WSL2 filesystem.
    
    Args:
      file_path: Path to the file to read
    
    Returns:
      File contents as string
    
    Raises:
      FileNotFoundError: If file doesn't exist
      PermissionError: If file can't be read
    """
    result = await self.execute_command(f"cat '{file_path}'")
    
    if not result['success']:
      stderr = result['stderr'].lower()
      if 'no such file' in stderr or 'not found' in stderr:
        raise FileNotFoundError(f"File not found: {file_path}")
      elif 'permission denied' in stderr:
        raise PermissionError(f"Permission denied reading file: {file_path}")
      else:
        raise IOError(f"Error reading file {file_path}: {result['stderr']}")
    
    return result['stdout']
  
  @override
  async def write_file(self, file_path: str, content: str) -> bool:
    """Write content to a file in WSL2 filesystem.
    
    Args:
      file_path: Path to the file to write
      content: Content to write
    
    Returns:
      True if successful, False otherwise
    """
    # Escape content for shell
    escaped_content = content.replace("'", "'\"'\"'")
    
    # Create parent directory if needed
    dir_result = await self.execute_command(f"mkdir -p \"$(dirname '{file_path}')\"")
    if not dir_result['success']:
      return False
    
    # Write file
    result = await self.execute_command(f"echo '{escaped_content}' > '{file_path}'")
    return result['success']
  
  @override
  async def file_exists(self, file_path: str) -> bool:
    """Check if a file exists in WSL2 filesystem.
    
    Args:
      file_path: Path to check
    
    Returns:
      True if file exists, False otherwise
    """
    result = await self.execute_command(f"test -e '{file_path}'")
    return result['success']
  
  @override
  async def create_directory(self, dir_path: str) -> bool:
    """Create a directory in WSL2 filesystem.
    
    Args:
      dir_path: Path of directory to create
    
    Returns:
      True if successful, False otherwise
    """
    result = await self.execute_command(f"mkdir -p '{dir_path}'")
    return result['success']
  
  @override
  async def list_directory(self, dir_path: str) -> Optional[list[str]]:
    """List contents of a directory in WSL2 filesystem.
    
    Args:
      dir_path: Path of directory to list
    
    Returns:
      List of file/directory names, or None if failed
    """
    result = await self.execute_command(f"ls -1 '{dir_path}' 2>/dev/null")
    if result['success'] and result['stdout'].strip():
      return result['stdout'].strip().split('\n')
    return []
  
  @override
  async def remove_file(self, file_path: str) -> bool:
    """Remove a file from WSL2 filesystem.
    
    Args:
      file_path: Path to the file to remove
    
    Returns:
      True if successful, False otherwise
    """
    result = await self.execute_command(f"rm -f '{file_path}'")
    return result['success']
  
  @override
  async def copy_file(self, source: str, destination: str) -> bool:
    """Copy a file within WSL2 filesystem.
    
    Args:
      source: Source file path
      destination: Destination file path
    
    Returns:
      True if successful, False otherwise
    """
    # Create destination directory if needed
    dir_result = await self.execute_command(f"mkdir -p \"$(dirname '{destination}')\"")
    if not dir_result['success']:
      return False
    
    result = await self.execute_command(f"cp '{source}' '{destination}'")
    return result['success']
  
  @override
  async def get_working_directory(self) -> Optional[str]:
    """Get the current working directory in WSL2.
    
    Returns:
      Current working directory path, or None if failed
    """
    if self._working_dir:
      return self._working_dir
    
    result = await self.execute_command("pwd")
    if result['success']:
      return result['stdout'].strip()
    return None
  
  @override
  async def set_working_directory(self, dir_path: str) -> bool:
    """Set the working directory in WSL2.
    
    Args:
      dir_path: Directory path to set as working directory
    
    Returns:
      True if successful, False otherwise
    """
    # Check if directory exists
    if not await self.file_exists(dir_path):
      return False
    
    # Test if it's a directory
    result = await self.execute_command(f"test -d '{dir_path}'")
    if not result['success']:
      return False
    
    self._working_dir = dir_path
    return True
  
  def get_available_distributions(self) -> list[str]:
    """Get list of available WSL distributions.
    
    Returns:
      List of distribution names
    """
    try:
      result = subprocess.run(
          ['wsl', '--list', '--quiet'],
          capture_output=True,
          text=True,
          timeout=5
      )
      
      if result.returncode == 0:
        distributions = result.stdout.strip().split('\n')
        return [d.strip() for d in distributions if d.strip()]
      
      return []
      
    except Exception:
      return []
  
  @override
  def get_info(self) -> Dict[str, Any]:
    """Get information about the WSL environment.
    
    Returns:
      Dictionary with environment information
    """
    info = super().get_info()
    info.update({
        'distribution': self.distribution or 'default',
        'available_distributions': self.get_available_distributions(),
        'working_directory': self._working_dir
    })
    return info
  
  @override
  def detect_capabilities(self) -> Dict[str, bool]:
    """Detect capabilities of the WSL environment.
    
    Returns:
      Dictionary of capability flags
    """
    capabilities = super().detect_capabilities()
    
    # This would need to be implemented by actually checking in WSL
    # For now, assume common Linux tools are available
    capabilities.update({
        'python_available': True,  # Most WSL distributions have Python
        'docker_available': False,  # Docker in WSL requires special setup
        'git_available': True,     # Git is commonly available
        'node_available': False,   # Node.js may not be installed
        'bash_available': True,    # WSL always has bash
        'linux_tools': True,       # Standard Linux command-line tools
    })
    
    return capabilities

