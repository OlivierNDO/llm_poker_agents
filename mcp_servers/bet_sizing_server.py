"""
bet_sizing_server.py

JSON-RPC server for poker bet sizing analysis.
Wraps the BetSizingAnalyzer functionality.
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.agent_utils import BetSizingAnalyzer
    from src.poker_types import ActionType
    import src.feature_engineering as fe
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


class TableConfigData:
    """Simple data class to hold table configuration"""
    def __init__(self, small_blind: int = 1, big_blind: int = 2):
        self.small_blind = small_blind
        self.big_blind = big_blind


class MockPlayer:
    """Mock player object for BetSizingAnalyzer"""
    def __init__(self, chips: int, current_bet: int = 0):
        self.chips = chips
        self.current_bet = current_bet


class MockGameState:
    """Mock game state object for BetSizingAnalyzer"""
    def __init__(self, pot: int, current_bet: int, betting_round: int = 1, 
                 active_players: int = 2, dealer_position: int = 0):
        self.pot = pot
        self.current_bet = current_bet
        self.betting_round = betting_round
        self.dealer_position = dealer_position
        self._active_players_count = active_players
        self._actions_since_board_change = []
    
    def get_active_players(self):
        """Return mock active players list"""
        return [None] * self._active_players_count
    
    def get_actions_since_board_change(self):
        """Return mock actions"""
        return self._actions_since_board_change


class BetSizingServer:
    """JSON-RPC server for bet sizing analysis"""
    
    def __init__(self):
        self.tools = {
            "get_all_sizing_context": {
                "description": "Get comprehensive bet/raise sizing analysis and recommendations",
                "method": self.get_all_sizing_context
            },
            "get_financial_context": {
                "description": "Get basic financial situation for betting",
                "method": self.get_financial_context
            },
            "get_pot_based_sizes": {
                "description": "Get standard pot-based sizing options",
                "method": self.get_pot_based_sizes
            },
            "get_blind_based_sizes": {
                "description": "Get big blind multiple sizing options",
                "method": self.get_blind_based_sizes
            },
            "get_stack_depth_context": {
                "description": "Get stack depth analysis relative to pot and blinds",
                "method": self.get_stack_depth_context
            },
            "get_position_context": {
                "description": "Get position-based sizing considerations",
                "method": self.get_position_context
            },
            "get_betting_round_context": {
                "description": "Get context about current betting round and aggression",
                "method": self.get_betting_round_context
            }
        }

    def _create_sizing_analyzer(self, player_chips: int, player_current_bet: int,
                               pot: int, current_bet: int, action_type: str,
                               small_blind: int = 1, big_blind: int = 2,
                               betting_round: int = 1, active_players: int = 2,
                               dealer_position: int = 0) -> BetSizingAnalyzer:
        """Create a BetSizingAnalyzer with mock objects"""
        
        # Convert action type string to ActionType enum
        action_type_map = {
            "bet": ActionType.BET,
            "raise": ActionType.RAISE,
            "BET": ActionType.BET,
            "RAISE": ActionType.RAISE
        }
        action_type_enum = action_type_map.get(action_type, ActionType.BET)
        
        # Create mock objects
        player = MockPlayer(player_chips, player_current_bet)
        game_state = MockGameState(pot, current_bet, betting_round, active_players, dealer_position)
        table_config = TableConfigData(small_blind, big_blind)
        
        return BetSizingAnalyzer(player, game_state, action_type_enum, table_config)

    def get_all_sizing_context(self, player_chips: int, player_current_bet: int,
                              pot: int, current_bet: int, action_type: str,
                              small_blind: int = 1, big_blind: int = 2,
                              betting_round: int = 1, active_players: int = 2,
                              dealer_position: int = 0) -> str:
        """Get comprehensive sizing analysis"""
        try:
            analyzer = self._create_sizing_analyzer(
                player_chips, player_current_bet, pot, current_bet, action_type,
                small_blind, big_blind, betting_round, active_players, dealer_position
            )
            
            context_strings = analyzer.get_all_sizing_context()
            return "\n".join(context_strings)
            
        except Exception as e:
            return f"Error in sizing analysis: {str(e)}"

    def get_financial_context(self, player_chips: int, player_current_bet: int,
                             pot: int, current_bet: int, action_type: str) -> str:
        """Get basic financial situation"""
        try:
            analyzer = self._create_sizing_analyzer(
                player_chips, player_current_bet, pot, current_bet, action_type
            )
            return analyzer.get_financial_context_string()
            
        except Exception as e:
            return f"Error in financial context: {str(e)}"

    def get_pot_based_sizes(self, pot: int, player_chips: int, 
                           player_current_bet: int, current_bet: int,
                           action_type: str) -> str:
        """Get pot-based sizing options"""
        try:
            analyzer = self._create_sizing_analyzer(
                player_chips, player_current_bet, pot, current_bet, action_type
            )
            return analyzer.get_pot_based_sizes_string()
            
        except Exception as e:
            return f"Error in pot-based sizes: {str(e)}"

    def get_blind_based_sizes(self, player_chips: int, player_current_bet: int,
                             pot: int, current_bet: int, action_type: str,
                             small_blind: int = 1, big_blind: int = 2) -> str:
        """Get blind-based sizing options"""
        try:
            analyzer = self._create_sizing_analyzer(
                player_chips, player_current_bet, pot, current_bet, action_type,
                small_blind, big_blind
            )
            return analyzer.get_blind_based_sizes_string()
            
        except Exception as e:
            return f"Error in blind-based sizes: {str(e)}"

    def get_stack_depth_context(self, player_chips: int, player_current_bet: int,
                               pot: int, current_bet: int, action_type: str,
                               small_blind: int = 1, big_blind: int = 2) -> str:
        """Get stack depth analysis"""
        try:
            analyzer = self._create_sizing_analyzer(
                player_chips, player_current_bet, pot, current_bet, action_type,
                small_blind, big_blind
            )
            return analyzer.get_stack_depth_context_string()
            
        except Exception as e:
            return f"Error in stack depth context: {str(e)}"

    def get_position_context(self, player_chips: int, player_current_bet: int,
                            pot: int, current_bet: int, action_type: str,
                            active_players: int = 2, dealer_position: int = 0) -> str:
        """Get position-based context"""
        try:
            analyzer = self._create_sizing_analyzer(
                player_chips, player_current_bet, pot, current_bet, action_type,
                active_players=active_players, dealer_position=dealer_position
            )
            return analyzer.get_position_context_string()
            
        except Exception as e:
            return f"Error in position context: {str(e)}"

    def get_betting_round_context(self, player_chips: int, player_current_bet: int,
                                 pot: int, current_bet: int, action_type: str,
                                 betting_round: int = 1) -> str:
        """Get betting round context"""
        try:
            analyzer = self._create_sizing_analyzer(
                player_chips, player_current_bet, pot, current_bet, action_type,
                betting_round=betting_round
            )
            return analyzer.get_betting_round_context_string()
            
        except Exception as e:
            return f"Error in betting round context: {str(e)}"

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request"""
        try:
            method = request.get("method")
            request_id = request.get("id")
            
            if method == "tools/list":
                # List available tools
                tools = []
                for name, info in self.tools.items():
                    tools.append({
                        "name": name,
                        "description": info["description"]
                    })
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": tools
                    }
                }
            
            elif method == "tools/call":
                # Call a specific tool
                params = request.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if tool_name not in self.tools:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {tool_name}"
                        }
                    }
                
                # Call the tool method
                tool_method = self.tools[tool_name]["method"]
                result_text = tool_method(**arguments)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"text": result_text}]
                    }
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
        
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    def run(self):
        """Run the server, reading from stdin and writing to stdout"""
        try:
            while True:
                # Read a line from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON request
                    request = json.loads(line)
                    
                    # Handle the request
                    response = self.handle_request(request)
                    
                    # Send JSON response
                    response_json = json.dumps(response)
                    print(response_json)
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    # Send error response for invalid JSON
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {str(e)}"
                        }
                    }
                    print(json.dumps(error_response))
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)


def main():
    """Main entry point"""
    server = BetSizingServer()
    server.run()


if __name__ == "__main__":
    main()