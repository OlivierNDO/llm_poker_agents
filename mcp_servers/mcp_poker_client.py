"""
mcp_poker_client.py - Windows-compatible version with proper timeouts
"""

import asyncio
import json
import os
import sys
import subprocess
import time
import threading
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class ServerConfig:
    """Configuration for an MCP server"""
    name: str
    script_path: str
    description: str

class WindowsCompatibleMCPClient:
    """Windows-compatible MCP client with proper timeout handling"""
    
    def __init__(self):
        self.servers = {}  # name -> subprocess.Popen
        self.server_configs = {}  # name -> ServerConfig
        
    def start_server(self, config: ServerConfig) -> bool:
        """Start an MCP server subprocess"""
        try:
            # Stop existing server if running
            if config.name in self.servers:
                self.stop_server(config.name)
            
            script_path = os.path.abspath(config.script_path)
            if not os.path.exists(script_path):
                print(f"Server script not found: {script_path}")
                return False
            
            script_dir = os.path.dirname(script_path)
            if os.path.basename(script_dir) == 'mcp_servers':
                working_dir = os.path.dirname(script_dir)
            else:
                working_dir = script_dir
            
            print(f"Starting server: {script_path}")
            print(f"Working directory: {working_dir}")
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir
            )
            
            self.servers[config.name] = process
            self.server_configs[config.name] = config
            print(f"Started server {config.name} (PID: {process.pid})")
            
            # Give server time to initialize
            time.sleep(2)
            
            # Check if it's still running
            if process.poll() is not None:
                print(f"Server {config.name} exited immediately with code: {process.poll()}")
                stderr_output = process.stderr.read()
                if stderr_output:
                    print(f"Server stderr: {stderr_output}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Failed to start server {config.name}: {e}")
            return False
    
    def stop_server(self, server_name: str):
        """Stop a server subprocess"""
        if server_name in self.servers:
            process = self.servers[server_name]
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            except Exception as e:
                print(f"Error stopping server {server_name}: {e}")
            
            del self.servers[server_name]
            print(f"Stopped server {server_name}")
    
    def stop_all_servers(self):
        """Stop all server subprocesses"""
        for server_name in list(self.servers.keys()):
            self.stop_server(server_name)
    
    def _read_with_timeout(self, process, timeout_seconds=15):
        """Read from process stdout with timeout using threading"""
        result = [None]  # Use list to allow modification from inner function
        exception = [None]
        
        def read_line():
            try:
                result[0] = process.stdout.readline()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=read_line)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Timeout occurred
            return None, "timeout"
        
        if exception[0]:
            return None, f"error: {exception[0]}"
        
        return result[0], None
    
    def send_request(self, server_name: str, request: dict) -> Optional[dict]:
        """Send a JSON-RPC request to a server with Windows-compatible timeout"""
        if server_name not in self.servers:
            print(f"Server {server_name} not found")
            return None
        
        process = self.servers[server_name]
        
        # Check if process is still alive
        if process.poll() is not None:
            print(f"Server {server_name} has died (exit code: {process.poll()})")
            return None
        
        try:
            # Send request
            request_json = json.dumps(request) + '\n'
            print(f"[CLIENT] Sending to {server_name}: {request_json.strip()}")
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # Read response with timeout
            response_line, error = self._read_with_timeout(process, timeout_seconds=15)
            
            if error == "timeout":
                print(f"[CLIENT] Timeout waiting for response from {server_name}")
                return None
            elif error:
                print(f"[CLIENT] Error reading from {server_name}: {error}")
                return None
            elif response_line:
                print(f"[CLIENT] Response from {server_name}: {response_line.strip()}")
                try:
                    return json.loads(response_line.strip())
                except json.JSONDecodeError as e:
                    print(f"[CLIENT] JSON decode error: {e}")
                    return None
            else:
                print(f"[CLIENT] Empty response from {server_name}")
                return None
            
        except Exception as e:
            print(f"Error communicating with server {server_name}: {e}")
            return None
    
    def list_tools(self, server_name: str) -> List[dict]:
        """List tools available from a server"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }
        
        response = self.send_request(server_name, request)
        if response and "result" in response:
            tools = response["result"].get("tools", [])
            print(f"[CLIENT] Tools from {server_name}: {[t.get('name') for t in tools]}")
            return tools
        return []
    
    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Optional[str]:
        """Call a tool on a server"""
        print(f"[CLIENT] Calling tool {tool_name} on {server_name} with args: {arguments}")
        
        request = {
            "jsonrpc": "2.0", 
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = self.send_request(server_name, request)
        if response and "result" in response:
            content = response["result"].get("content", [])
            if content and len(content) > 0:
                result_text = content[0].get("text", "")
                print(f"[CLIENT] Tool result: {result_text[:200]}...")
                return result_text
        elif response and "error" in response:
            print(f"[CLIENT] Tool call error: {response['error']}")
        
        return None

class MCPPokerClient:
    """High-level poker client that wraps WindowsCompatibleMCPClient"""
    
    def __init__(self):
        self.client = WindowsCompatibleMCPClient()
        self.connected_servers = []
        
    async def connect_to_hand_analysis_server(self, script_path: str = None) -> bool:
        """Connect to the hand analysis server"""
        if script_path is None:
            current_dir = os.path.dirname(__file__)
            script_path = os.path.join(current_dir, "hand_analysis_server.py")
        
        config = ServerConfig(
            name="hand_analysis",
            script_path=script_path,
            description="Poker hand analysis and probabilities"
        )
        
        success = self.client.start_server(config)
        if success:
            self.connected_servers.append("hand_analysis")
            
            # Test the connection by listing tools
            tools = self.client.list_tools("hand_analysis")
            if tools:
                print(f"Hand analysis server ready with {len(tools)} tools")
                return True
            else:
                print("Hand analysis server started but no tools found")
                self.client.stop_server("hand_analysis")
                if "hand_analysis" in self.connected_servers:
                    self.connected_servers.remove("hand_analysis")
                return False
        
        return False
    
    async def connect_to_predictive_model_server(self, script_path: str = None) -> bool:
        """Connect to the Monte Carlo predictive model server"""
        if script_path is None:
            current_dir = os.path.dirname(__file__)
            script_path = os.path.join(current_dir, "predictive_model_server.py")
        
        print(f"Attempting to connect to Monte Carlo server at: {script_path}")
        
        config = ServerConfig(
            name="predictive_model",
            script_path=script_path,
            description="Poker outcome predictions using Monte Carlo simulation"
        )
        
        success = self.client.start_server(config)
        if success:
            self.connected_servers.append("predictive_model")
            
            # Test the connection
            tools = self.client.list_tools("predictive_model")
            if tools:
                print(f"Monte Carlo server ready with {len(tools)} tools")
                print(f"Available tools: {[t.get('name') for t in tools]}")
                return True
            else:
                print("Monte Carlo server started but no tools found")
                return False
        
        return False
    
    async def predict_outcome_monte_carlo(self, hole_cards: List[str], board_cards: List[str], 
                                         num_players: int, iterations: int = 2500) -> Optional[dict]:
        """Get Monte Carlo-based outcome prediction"""
        print(f"[POKER_CLIENT] Monte Carlo prediction request: {hole_cards}, {board_cards}, {num_players}")
        
        result = self.client.call_tool("predictive_model", "predict_outcome_monte_carlo", {
            "hole_cards": hole_cards,
            "board_cards": board_cards,
            "num_players": num_players,
            "iterations": 2500
        })
        
        if result:
            try:
                parsed = json.loads(result)
                print(f"[POKER_CLIENT] Monte Carlo prediction result: {parsed}")
                return parsed
            except json.JSONDecodeError as e:
                print(f"[POKER_CLIENT] JSON decode error: {e}")
                print(f"[POKER_CLIENT] Raw result: {result}")
                return None
        
        print("[POKER_CLIENT] No result from Monte Carlo prediction")
        return None
    
    async def get_win_probability_fast(self, hole_cards: List[str], board_cards: List[str], num_players: int) -> Optional[float]:
        """Get quick win probability estimate"""
        result = self.client.call_tool("predictive_model", "get_win_probability", {
            "hole_cards": hole_cards,
            "board_cards": board_cards,
            "num_players": num_players
        })
        
        if result:
            try:
                return float(result.strip())
            except ValueError:
                return None
        return None
    
    async def get_hand_analysis(self, hole_cards: List[str], board_cards: List[str], num_opponents: int) -> Optional[str]:
        """Get comprehensive hand analysis"""
        return self.client.call_tool("hand_analysis", "get_hand_features", {
            "hole_cards": hole_cards,
            "board_cards": board_cards,
            "num_opponents": num_opponents
        })
    
    def get_available_capabilities(self) -> List[str]:
        """Get list of available capabilities"""
        capabilities = []
        
        if "predictive_model" in self.connected_servers:
            tools = self.client.list_tools("predictive_model")
            if any(tool.get("name") == "predict_outcome_monte_carlo" for tool in tools):
                capabilities.append("monte_carlo_simulation")
            if any(tool.get("name") == "get_win_probability" for tool in tools):
                capabilities.append("quick_win_probability")
        
        if "hand_analysis" in self.connected_servers:
            tools = self.client.list_tools("hand_analysis")
            if any(tool.get("name") == "get_hand_features" for tool in tools):
                capabilities.append("comprehensive_hand_analysis")
            
        return capabilities
    
    def is_connected(self, server_name: str) -> bool:
        """Check if connected to a specific server"""
        return server_name in self.connected_servers
    
    async def disconnect_all(self):
        """Disconnect from all servers"""
        self.client.stop_all_servers()
        self.connected_servers.clear()

# Test function
async def test_monte_carlo():
    """Test the Monte Carlo server connection"""
    try:
        client = MCPPokerClient()
        
        # Find the server script
        current_dir = os.path.dirname(__file__)
        server_script = os.path.join(current_dir, "predictive_model_server.py")
        
        if not os.path.exists(server_script):
            print(f"Server script not found at: {server_script}")
            return
        
        print("Connecting to Monte Carlo server...")
        success = await client.connect_to_predictive_model_server(server_script)
        
        if not success:
            print("Failed to connect to Monte Carlo server")
            return
        
        print("Connected successfully!")
        print("Available capabilities:", client.get_available_capabilities())
        
        # Test prediction
        hole_cards = ["As", "Kd"]
        board_cards = ["Ac", "4s", "9d"]
        num_players = 2
        
        print(f"\nTesting Monte Carlo prediction with {hole_cards} on {board_cards} vs {num_players} players:")
        
        result = await client.predict_outcome_monte_carlo(hole_cards, board_cards, num_players)
        print(f"Monte Carlo prediction result: {result}")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            await client.disconnect_all()

if __name__ == "__main__":
    asyncio.run(test_monte_carlo())