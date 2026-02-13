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

"""SSH environment implementation for remote execution."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from typing import Dict
from typing import Optional

from typing_extensions import override

from .base_environment import BaseEnvironment


class SSHEnvironment(BaseEnvironment):
  """SSH execution environment.
  
  This environment executes commands and performs file operations on
  remote machines via SSH. Note: This is a simplified implementation
  that uses command-line SSH. A production version would use paramiko
  or similar SSH library for better connection management.
  """
  
  def __init__(
      self,
      host: str,
      username: str,
      password: Optional[str] = None,
      key_file: Optional[str] = None,
      port: int = 22
  ):
    """Initialize the SSH environment.
    
    Args:
      host: SSH host address
      username: SSH username
      password: SSH password (if not using key)
      key_file: Path to SSH private key file
      port: SSH port (default 22)
    """
    super().__init__(f"ssh_{host}")
    self.host = host
    self.username = username
    self.password = password
    self.key_file = key_file
    self.port = port
    self._working_dir: Optional[str] = None
  
  def _build_ssh_command(self, command: str) -> list[str]:
    """Build SSH command with proper authentication.
    
    Args:
      command: Command to execute remotely
    
    Returns:
      List of command parts for subprocess
    """
    ssh_cmd = ['ssh']
    
    # Add port if not default
    if self.port != 22:
      ssh_cmd.extend(['-p', str(self.port)])
    
    # Add key file if specified
    if self.key_file:
      ssh_cmd.extend(['-i', self.key_file])
    
    # Disable host key checking for simplicity (not recommended for production)
    ssh_cmd.extend(['-o', 'StrictHostKeyChecking=no'])
    
    # Add user@host
    ssh_cmd.append(f"{self.username}@{self.host}")
    
    # Add command
    ssh_cmd.append(command)
    
    return ssh_cmd
  
  @override
  def is_available(self) -> bool:
    """Check if SSH connection is available.
    
    Returns:
      True if SSH connection can be established, False otherwise
    """
    try:
      # Simple connectivity test
      import subprocess
      
      test_cmd = self._build_ssh_command('echo "test"')
      result = subprocess.run(
          test_cmd,
          capture_output=True,
          timeout=10
      )
      
      return result.returncode == 0
      
    except Exception:
      return False
  
  @override
  async def execute_command(
      self,
      command: str,
      timeout: Optional[float] = None,
      working_dir: Optional[str] = None
  ) -> Dict[str, Any]:
    """Execute a command via SSH.
    
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
      
      # Build SSH command
      ssh_cmd = self._build_ssh_command(command)
      
      # Create subprocess
      process = await asyncio.create_subprocess_exec(
          *ssh_cmd,
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
    """Read a file from remote filesystem via SSH.
    
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
    """Write content to a file on remote filesystem via SSH.
    
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
    """Check if a file exists on remote filesystem via SSH.
    
    Args:
      file_path: Path to check
    
    Returns:
      True if file exists, False otherwise
    """
    result = await self.execute_command(f"test -e '{file_path}'")
    return result['success']
  
  def close(self) -> None:
    """Close SSH connection.
    
    Note: In this simplified implementation, each command creates
    its own connection. A production version would maintain persistent
    connections that need to be closed.
    """
    # In a real implementation with persistent connections,
    # this would close the SSH connection
    pass
  
  @override
  def get_info(self) -> Dict[str, Any]:
    """Get information about the SSH environment.
    
    Returns:
      Dictionary with environment information
    """
    info = super().get_info()
    info.update({
        'host': self.host,
        'username': self.username,
        'port': self.port,
        'uses_key': self.key_file is not None,
        'working_directory': self._working_dir
    })
    return info
  
  @override
  def detect_capabilities(self) -> Dict[str, bool]:
    """Detect capabilities of the SSH environment.
    
    Returns:
      Dictionary of capability flags
    """
    capabilities = super().detect_capabilities()
    
    # This would need to be implemented by actually checking on the remote host
    # For now, assume basic Unix/Linux capabilities
    capabilities.update({
        'python_available': False,  # Would need to check
        'docker_available': False,  # Would need to check
        'git_available': False,     # Would need to check
        'node_available': False,    # Would need to check
        'bash_available': True,     # Most SSH hosts have bash
        'unix_tools': True,         # Basic Unix tools usually available
    })
    
    return capabilities

