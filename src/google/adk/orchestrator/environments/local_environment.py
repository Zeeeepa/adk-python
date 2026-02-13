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

"""Local environment implementation for direct local execution."""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from typing_extensions import override

from .base_environment import BaseEnvironment


class LocalEnvironment(BaseEnvironment):
  """Local execution environment.
  
  This environment executes commands and performs file operations directly
  on the local machine where the orchestrator is running.
  """
  
  def __init__(self):
    """Initialize the local environment."""
    super().__init__("local")
    self._working_dir: Optional[str] = None
  
  @override
  def is_available(self) -> bool:
    """Check if the local environment is available.
    
    Returns:
      Always True for local environment
    """
    return True
  
  @override
  async def execute_command(
      self,
      command: str,
      timeout: Optional[float] = None,
      working_dir: Optional[str] = None
  ) -> Dict[str, Any]:
    """Execute a command locally.
    
    Args:
      command: Command to execute
      timeout: Command timeout in seconds
      working_dir: Working directory for the command
    
    Returns:
      Dictionary with execution results
    """
    start_time = time.time()
    
    try:
      # Determine working directory
      cwd = working_dir or self._working_dir or os.getcwd()
      
      # Create subprocess
      process = await asyncio.create_subprocess_shell(
          command,
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE,
          cwd=cwd
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
    """Read a file from the local filesystem.
    
    Args:
      file_path: Path to the file to read
    
    Returns:
      File contents as string
    
    Raises:
      FileNotFoundError: If file doesn't exist
      PermissionError: If file can't be read
    """
    try:
      path = Path(file_path)
      return await asyncio.to_thread(path.read_text, encoding='utf-8')
    except Exception as e:
      if isinstance(e, FileNotFoundError):
        raise FileNotFoundError(f"File not found: {file_path}")
      elif isinstance(e, PermissionError):
        raise PermissionError(f"Permission denied reading file: {file_path}")
      else:
        raise IOError(f"Error reading file {file_path}: {e}")
  
  @override
  async def write_file(self, file_path: str, content: str) -> bool:
    """Write content to a file on the local filesystem.
    
    Args:
      file_path: Path to the file to write
      content: Content to write
    
    Returns:
      True if successful, False otherwise
    """
    try:
      path = Path(file_path)
      # Create parent directories if they don't exist
      path.parent.mkdir(parents=True, exist_ok=True)
      await asyncio.to_thread(path.write_text, content, encoding='utf-8')
      return True
    except Exception:
      return False
  
  @override
  async def file_exists(self, file_path: str) -> bool:
    """Check if a file exists on the local filesystem.
    
    Args:
      file_path: Path to check
    
    Returns:
      True if file exists, False otherwise
    """
    try:
      path = Path(file_path)
      return await asyncio.to_thread(path.exists)
    except Exception:
      return False
  
  @override
  async def create_directory(self, dir_path: str) -> bool:
    """Create a directory on the local filesystem.
    
    Args:
      dir_path: Path of directory to create
    
    Returns:
      True if successful, False otherwise
    """
    try:
      path = Path(dir_path)
      await asyncio.to_thread(path.mkdir, parents=True, exist_ok=True)
      return True
    except Exception:
      return False
  
  @override
  async def list_directory(self, dir_path: str) -> Optional[list[str]]:
    """List contents of a directory on the local filesystem.
    
    Args:
      dir_path: Path of directory to list
    
    Returns:
      List of file/directory names, or None if failed
    """
    try:
      path = Path(dir_path)
      if not await asyncio.to_thread(path.exists):
        return None
      
      if not await asyncio.to_thread(path.is_dir):
        return None
      
      entries = await asyncio.to_thread(list, path.iterdir())
      return [entry.name for entry in entries]
    except Exception:
      return None
  
  @override
  async def remove_file(self, file_path: str) -> bool:
    """Remove a file from the local filesystem.
    
    Args:
      file_path: Path to the file to remove
    
    Returns:
      True if successful, False otherwise
    """
    try:
      path = Path(file_path)
      if await asyncio.to_thread(path.exists):
        await asyncio.to_thread(path.unlink)
      return True
    except Exception:
      return False
  
  @override
  async def copy_file(self, source: str, destination: str) -> bool:
    """Copy a file on the local filesystem.
    
    Args:
      source: Source file path
      destination: Destination file path
    
    Returns:
      True if successful, False otherwise
    """
    try:
      source_path = Path(source)
      dest_path = Path(destination)
      
      # Create destination directory if needed
      dest_path.parent.mkdir(parents=True, exist_ok=True)
      
      await asyncio.to_thread(shutil.copy2, source_path, dest_path)
      return True
    except Exception:
      return False
  
  @override
  async def get_working_directory(self) -> Optional[str]:
    """Get the current working directory.
    
    Returns:
      Current working directory path
    """
    try:
      return self._working_dir or os.getcwd()
    except Exception:
      return None
  
  @override
  async def set_working_directory(self, dir_path: str) -> bool:
    """Set the working directory.
    
    Args:
      dir_path: Directory path to set as working directory
    
    Returns:
      True if successful, False otherwise
    """
    try:
      path = Path(dir_path)
      if await asyncio.to_thread(path.exists) and await asyncio.to_thread(path.is_dir):
        self._working_dir = str(path.resolve())
        return True
      return False
    except Exception:
      return False
  
  @override
  def get_info(self) -> Dict[str, Any]:
    """Get information about the local environment.
    
    Returns:
      Dictionary with environment information
    """
    info = super().get_info()
    info.update({
        'platform': platform.system(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'python_version': platform.python_version(),
        'working_directory': self._working_dir or os.getcwd()
    })
    return info
  
  @override
  def detect_capabilities(self) -> Dict[str, bool]:
    """Detect capabilities of the local environment.
    
    Returns:
      Dictionary of capability flags
    """
    capabilities = super().detect_capabilities()
    
    # Check for common tools
    capabilities.update({
        'python_available': shutil.which('python') is not None or shutil.which('python3') is not None,
        'docker_available': shutil.which('docker') is not None,
        'git_available': shutil.which('git') is not None,
        'node_available': shutil.which('node') is not None,
        'npm_available': shutil.which('npm') is not None,
        'pip_available': shutil.which('pip') is not None or shutil.which('pip3') is not None,
    })
    
    return capabilities

