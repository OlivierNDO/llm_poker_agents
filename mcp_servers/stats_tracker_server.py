"""
stats_tracker_server.py

JSON-RPC server for poker statistics tracking and analysis.
Wraps StatsTracker functionality to provide opponent analysis.
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.game import StatsTracker, PlayerStats
    from src.poker_types import ActionType, Action
    import src.feature_engineering as fe
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


class MockAction:
    """Mock action for stats tracking"""
    def __init__(self, action_type: str, amount: int = 0, reasons: Optional[List[str]] = None):
        # Convert string to ActionType enum
        action_type_map = {
            "fold": ActionType.FOLD,
            "check": ActionType.CHECK,
            "call": ActionType.CALL,
            "bet": ActionType.BET,
            "raise": ActionType.RAISE,
            "all_in": ActionType.ALL_IN
        }
        self.action_type = action_type_map.get(action_type.lower(), ActionType.FOLD)
        self.amount = amount
        self.reasons = reasons or []


class MockPlayer:
    """Mock player for stats tracking"""
    def __init__(self, name: str, chips: int = 1000):
        self.name = name
        self.chips = chips


class StatsTrackerServer:
    """JSON-RPC server for stats tracking and analysis"""
    
    def __init__(self):
        # In-memory stats tracker - in production this would be persistent
        self.stats_tracker = StatsTracker()
        
        self.tools = {
            "record_hand_start": {
                "description": "Record that a new hand has started for players",
                "method": self.record_hand_start
            },
            "record_hand_end": {
                "description": "Record hand results and winners",
                "method": self.record_hand_end
            },
            "record_action": {
                "description": "Record a player action for statistics",
                "method": self.record_action
            },
            "get_opponent_stats_summary": {
                "description": "Get formatted summary of opponent statistics",
                "method": self.get_opponent_stats_summary
            },
            "get_player_stats": {
                "description": "Get detailed statistics for a specific player",
                "method": self.get_player_stats
            },
            "get_player_tendencies": {
                "description": "Get behavioral tendencies analysis for a player",
                "method": self.get_player_tendencies
            },
            "get_table_dynamics": {
                "description": "Get overall table dynamics and player types",
                "method": self.get_table_dynamics
            },
            "analyze_player_vs_position": {
                "description": "Analyze how a player behaves in different positions",
                "method": self.analyze_player_vs_position
            }
        }

    def record_hand_start(self, player_names: List[str]) -> str:
        """Record that a hand has started"""
        try:
            players = [MockPlayer(name) for name in player_names]
            self.stats_tracker.record_hand_start(players)
            return f"Recorded hand start for {len(player_names)} players"
        except Exception as e:
            return f"Error recording hand start: {str(e)}"

    def record_hand_end(self, winner_names: List[str], 
                       showdown_player_names: List[str] = None) -> str:
        """Record hand end results"""
        try:
            winners = [MockPlayer(name) for name in winner_names]
            showdown_players = [MockPlayer(name) for name in (showdown_player_names or [])]
            
            self.stats_tracker.record_hand_end(winners, showdown_players)
            return f"Recorded hand end: {len(winner_names)} winners, {len(showdown_players)} in showdown"
        except Exception as e:
            return f"Error recording hand end: {str(e)}"

    def record_action(self, player_name: str, action_type: str, 
                     betting_round: int, voluntary: bool = True,
                     amount: int = 0, reasons: List[str] = None) -> str:
        """Record a player action"""
        try:
            action = MockAction(action_type, amount, reasons)
            self.stats_tracker.record_action(player_name, action, betting_round, voluntary)
            return f"Recorded {action_type} action for {player_name}"
        except Exception as e:
            return f"Error recording action: {str(e)}"

    def get_opponent_stats_summary(self, my_name: str, 
                                  include_win_rates: bool = False) -> str:
        """Get opponent statistics summary"""
        try:
            return self.stats_tracker.get_opponent_stats_summary(my_name, include_win_rates)
        except Exception as e:
            return f"Error getting opponent stats: {str(e)}"

    def get_player_stats(self, player_name: str) -> str:
        """Get detailed stats for a specific player"""
        try:
            stats = self.stats_tracker.get_player_stats(player_name)
            
            lines = []
            lines.append(f"=== {player_name} Statistics ===")
            lines.append(f"Hands played: {stats.hands_played}")
            lines.append(f"Hands won: {stats.hands_won} ({stats.get_win_rate():.1f}%)")
            lines.append(f"VPIP: {stats.get_vpip_percent():.1f}%")
            lines.append(f"PFR: {stats.get_pfr_percent():.1f}%")
            lines.append(f"Aggression Factor: {stats.get_aggression_factor():.1f}")
            lines.append(f"WTSD: {stats.get_wtsd():.1f}%")
            
            # Action breakdown
            total_actions = (stats.total_bets + stats.total_calls + 
                           stats.total_raises + stats.total_checks + stats.total_folds)
            if total_actions > 0:
                lines.append("")
                lines.append("Action Distribution:")
                lines.append(f"  Bets: {stats.total_bets} ({stats.total_bets/total_actions*100:.1f}%)")
                lines.append(f"  Calls: {stats.total_calls} ({stats.total_calls/total_actions*100:.1f}%)")
                lines.append(f"  Raises: {stats.total_raises} ({stats.total_raises/total_actions*100:.1f}%)")
                lines.append(f"  Checks: {stats.total_checks} ({stats.total_checks/total_actions*100:.1f}%)")
                lines.append(f"  Folds: {stats.total_folds} ({stats.total_folds/total_actions*100:.1f}%)")
            
            # Sample size warning
            if stats.hands_played < 20:
                lines.append("")
                lines.append(f"WARNING: Small sample size ({stats.hands_played} hands)")
                lines.append("Statistics may not be reliable")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error getting player stats: {str(e)}"

    def get_player_tendencies(self, player_name: str) -> str:
        """Analyze player behavioral tendencies"""
        try:
            stats = self.stats_tracker.get_player_stats(player_name)
            
            if stats.hands_played == 0:
                return f"No data available for {player_name}"
            
            lines = []
            lines.append(f"=== {player_name} Behavioral Analysis ===")
            
            # Playing style classification
            vpip = stats.get_vpip_percent()
            pfr = stats.get_pfr_percent()
            agg = stats.get_aggression_factor()
            
            # Tightness/Looseness
            if vpip < 15:
                tightness = "Very Tight"
            elif vpip < 23:
                tightness = "Tight"
            elif vpip < 35:
                tightness = "Loose"
            else:
                tightness = "Very Loose"
            
            # Aggressiveness
            if agg < 1.0:
                aggressiveness = "Passive"
            elif agg < 2.0:
                aggressiveness = "Moderate"
            elif agg < 4.0:
                aggressiveness = "Aggressive"
            else:
                aggressiveness = "Very Aggressive"
            
            lines.append(f"Playing Style: {tightness} {aggressiveness}")
            lines.append(f"VPIP: {vpip:.1f}% | PFR: {pfr:.1f}% | AGG: {agg:.1f}")
            
            # Tendencies
            lines.append("")
            lines.append("Key Tendencies:")
            
            if vpip > 30:
                lines.append("- Plays many hands, likely recreational player")
            elif vpip < 15:
                lines.append("- Very selective with hands, wait for premium spots")
            
            if pfr / vpip > 0.7 if vpip > 0 else False:
                lines.append("- Aggressive preflop, likely raises with most playable hands")
            elif pfr / vpip < 0.3 if vpip > 0 else False:
                lines.append("- Often limps/calls preflop, avoid bluffing")
            
            if agg > 3.0:
                lines.append("- Highly aggressive post-flop, be cautious of bluffs")
            elif agg < 1.0:
                lines.append("- Passive post-flop, bets/raises likely indicate strength")
            
            # Showdown tendencies
            wtsd = stats.get_wtsd()
            if wtsd > 30:
                lines.append("- Goes to showdown frequently, likely calls light")
            elif wtsd < 15:
                lines.append("- Avoids showdowns, likely folds marginal hands")
            
            # Sample size warning
            if stats.hands_played < 30:
                lines.append("")
                lines.append(f"Note: Based on only {stats.hands_played} hands - use with caution")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error analyzing player tendencies: {str(e)}"

    def get_table_dynamics(self, player_names: List[str]) -> str:
        """Analyze overall table dynamics"""
        try:
            if not player_names:
                return "No players specified for table analysis"
            
            lines = []
            lines.append("=== Table Dynamics Analysis ===")
            
            player_types = []
            total_vpip = 0
            total_agg = 0
            valid_players = 0
            
            for name in player_names:
                stats = self.stats_tracker.get_player_stats(name)
                if stats.hands_played > 0:
                    vpip = stats.get_vpip_percent()
                    agg = stats.get_aggression_factor()
                    
                    total_vpip += vpip
                    total_agg += agg
                    valid_players += 1
                    
                    # Classify player type
                    if vpip < 20 and agg > 2.0:
                        player_type = "TAG (Tight-Aggressive)"
                    elif vpip < 20 and agg < 1.5:
                        player_type = "Tight-Passive"
                    elif vpip > 30 and agg > 2.0:
                        player_type = "LAG (Loose-Aggressive)"
                    elif vpip > 30 and agg < 1.5:
                        player_type = "Loose-Passive"
                    else:
                        player_type = "Balanced"
                    
                    player_types.append(f"{name}: {player_type} (VPIP: {vpip:.1f}%, AGG: {agg:.1f})")
            
            if valid_players == 0:
                return "No player data available for table analysis"
            
            # Overall table characteristics
            avg_vpip = total_vpip / valid_players
            avg_agg = total_agg / valid_players
            
            lines.append(f"Players analyzed: {valid_players}")
            lines.append(f"Average VPIP: {avg_vpip:.1f}%")
            lines.append(f"Average Aggression: {avg_agg:.1f}")
            lines.append("")
            
            # Table classification
            if avg_vpip > 25:
                table_tightness = "Loose"
            elif avg_vpip < 18:
                table_tightness = "Tight"
            else:
                table_tightness = "Standard"
            
            if avg_agg > 2.5:
                table_aggression = "Aggressive"
            elif avg_agg < 1.5:
                table_aggression = "Passive"
            else:
                table_aggression = "Balanced"
            
            lines.append(f"Table Type: {table_tightness} {table_aggression}")
            lines.append("")
            
            # Strategic recommendations
            lines.append("Strategic Recommendations:")
            if table_tightness == "Tight":
                lines.append("- Steal blinds more frequently")
                lines.append("- Value bet thinner")
            elif table_tightness == "Loose":
                lines.append("- Tighten up hand selection")
                lines.append("- Value bet wider for profit")
            
            if table_aggression == "Aggressive":
                lines.append("- Be prepared for more bluffs")
                lines.append("- Call down lighter with medium strength hands")
            elif table_aggression == "Passive":
                lines.append("- Bluff more frequently")
                lines.append("- Respect bets and raises more")
            
            lines.append("")
            lines.append("Individual Player Types:")
            lines.extend(player_types)
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error analyzing table dynamics: {str(e)}"

    def analyze_player_vs_position(self, player_name: str) -> str:
        """Analyze how player behaves in different positions"""
        try:
            stats = self.stats_tracker.get_player_stats(player_name)
            
            if stats.hands_played < 10:
                return f"Insufficient data for positional analysis of {player_name} ({stats.hands_played} hands)"
            
            lines = []
            lines.append(f"=== {player_name} Positional Analysis ===")
            lines.append("Note: This is a basic analysis - full positional tracking requires enhanced data collection")
            lines.append("")
            
            # Basic analysis based on overall stats
            vpip = stats.get_vpip_percent()
            pfr = stats.get_pfr_percent()
            
            lines.append("Estimated Positional Tendencies:")
            
            # Early position estimation
            estimated_ep_vpip = max(5, vpip * 0.6)  # Tighter in early position
            lines.append(f"Early Position: ~{estimated_ep_vpip:.1f}% VPIP (estimated)")
            
            # Late position estimation  
            estimated_lp_vpip = min(50, vpip * 1.4)  # Looser in late position
            lines.append(f"Late Position: ~{estimated_lp_vpip:.1f}% VPIP (estimated)")
            
            # Button estimation
            estimated_btn_vpip = min(60, vpip * 1.6)  # Loosest on button
            lines.append(f"Button: ~{estimated_btn_vpip:.1f}% VPIP (estimated)")
            
            lines.append("")
            lines.append("Recommendations:")
            
            if estimated_lp_vpip - estimated_ep_vpip > 15:
                lines.append("- Player likely adjusts significantly by position")
                lines.append("- Respect early position opens more")
                lines.append("- Defend wider against late position opens")
            else:
                lines.append("- Player seems position-unaware or plays similar ranges")
                lines.append("- Treat opens from all positions more equally")
            
            lines.append("")
            lines.append("Note: For detailed positional analysis, implement position tracking in action recording")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error analyzing positional play: {str(e)}"

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
    server = StatsTrackerServer()
    server.run()


if __name__ == "__main__":
    main()