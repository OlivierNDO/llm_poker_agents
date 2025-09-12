#!/usr/bin/env python3
"""
game_state_server.py

JSON-RPC server for poker game state information.
Provides formatted game information and context.
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.poker_types import ActionType
    import src.feature_engineering as fe
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


class GameStateServer:
    """JSON-RPC server for game state information"""
    
    def __init__(self):
        self.tools = {
            "get_basic_game_info": {
                "description": "Get formatted basic game information",
                "method": self.get_basic_game_info
            },
            "get_pot_odds_info": {
                "description": "Calculate and format pot odds information",
                "method": self.get_pot_odds_info
            },
            "get_position_info": {
                "description": "Get position-based information and guidance",
                "method": self.get_position_info
            },
            "get_action_history_summary": {
                "description": "Get formatted summary of recent actions",
                "method": self.get_action_history_summary
            },
            "get_opponent_status_summary": {
                "description": "Get summary of all opponents' chip stacks and positions",
                "method": self.get_opponent_status_summary
            },
            "get_betting_round_summary": {
                "description": "Get summary of current betting round",
                "method": self.get_betting_round_summary
            },
            "check_blind_defense_situation": {
                "description": "Check if this is a blind defense situation and provide guidance",
                "method": self.check_blind_defense_situation
            }
        }

    def get_basic_game_info(self, player_name: str, player_chips: int, 
                           hole_cards: List[str], board_cards: List[str],
                           pot: int, betting_round: str, current_bet: int,
                           player_current_bet: int) -> str:
        """Get formatted basic game information"""
        
        lines = []
        lines.append(f"You are {player_name}, competing in a Texas Hold'em hand.")
        lines.append("All opponents are AI agents with similar information access.")
        lines.append("Objective: Maximize chip winnings over many hands.")
        
        # Cards and chips
        hole_str = " ".join(hole_cards) if hole_cards else "None"
        lines.append(f"Your hole cards: {hole_str}")
        lines.append(f"Your chips: {player_chips}")
        lines.append(f"Current pot: {pot}")
        
        # Board
        if board_cards:
            board_str = " ".join(board_cards)
            lines.append(f"Community cards: {board_str}")
        else:
            lines.append("Community cards: None dealt yet")
        
        # Betting situation
        lines.append(f"Betting round: {betting_round}")
        amount_to_call = max(0, current_bet - player_current_bet)
        
        if amount_to_call > 0:
            lines.append(f"You need to call {amount_to_call} to stay in the hand")
        else:
            lines.append("No bet to call - you can check or bet")
        
        lines.append(f"Your current bet this round: {player_current_bet}")
        
        return "\n".join(lines)

    def get_pot_odds_info(self, pot: int, amount_to_call: int, 
                         player_current_bet: int = 0) -> str:
        """Calculate and format pot odds information"""
        
        if amount_to_call <= 0:
            return "No call required - pot odds not applicable"
        
        # Calculate pot odds
        pot_odds_ratio = pot / amount_to_call
        break_even_equity = amount_to_call / (pot + amount_to_call) * 100
        
        lines = []
        lines.append(f"Pot odds: {pot_odds_ratio:.1f}:1")
        lines.append(f"Break-even equity needed: {break_even_equity:.1f}%")
        
        # Add guidance based on pot odds
        if pot_odds_ratio >= 3.0:
            lines.append("Excellent pot odds - consider calling with weak hands")
        elif pot_odds_ratio >= 2.0:
            lines.append("Good pot odds - favorable for drawing hands")
        elif pot_odds_ratio >= 1.5:
            lines.append("Decent pot odds - need reasonable hand strength")
        else:
            lines.append("Poor pot odds - need strong hand to call")
        
        # Blind defense context
        if player_current_bet > 0:
            lines.append(f"Note: You have {player_current_bet} chips already invested")
            if pot_odds_ratio >= 2.5:
                lines.append("BLIND DEFENSE: Getting excellent odds - fold only trash hands")
            elif pot_odds_ratio >= 2.0:
                lines.append("BLIND DEFENSE: Getting good odds - defend reasonable range")
        
        return "\n".join(lines)

    def get_position_info(self, position: str, active_players: int, 
                         dealer_position: int = 0, betting_round: str = "PREFLOP") -> str:
        """Get position-based information"""
        
        lines = []
        
        # Position description
        if active_players == 2:
            lines.append("Position: Heads-up play")
        elif position == "dealer" or position == "button":
            lines.append("Position: Dealer/Button (best position)")
        elif position == "small_blind":
            lines.append("Position: Small Blind (out of position post-flop)")
        elif position == "big_blind":
            lines.append("Position: Big Blind (last to act preflop)")
        elif position == "early":
            lines.append("Position: Early position (tight play recommended)")
        elif position == "middle":
            lines.append("Position: Middle position (moderate aggression)")
        elif position == "late":
            lines.append("Position: Late position (can play wider range)")
        else:
            lines.append(f"Position: {position} vs {active_players-1} opponent(s)")
        
        # Position-specific guidance
        if betting_round == "PREFLOP":
            if position in ["dealer", "button", "late"]:
                lines.append("Preflop: Can play wider range, consider stealing blinds")
            elif position in ["small_blind", "big_blind"]:
                lines.append("Preflop: Defend against small raises with pot odds")
            else:
                lines.append("Preflop: Play tight, avoid marginal hands")
        else:
            if position in ["dealer", "button", "late"]:
                lines.append("Post-flop: Act last - can control pot size and bluff effectively")
            else:
                lines.append("Post-flop: Act first - check-call or bet for value/protection")
        
        return "\n".join(lines)

    def get_action_history_summary(self, actions: List[Dict[str, Any]], 
                                  current_round: str = "PREFLOP") -> str:
        """Get formatted action history summary"""
        
        if not actions:
            return f"No actions yet this {current_round.lower()} round"
        
        lines = []
        lines.append(f"Actions this {current_round.lower()} round:")
        
        # Group actions by player for clarity
        for action in actions:
            player = action.get("player", "Unknown")
            action_type = action.get("action", "unknown")
            amount = action.get("amount", 0)
            
            if amount and amount > 0:
                lines.append(f"  {player}: {action_type} {amount}")
            else:
                lines.append(f"  {player}: {action_type}")
        
        # Analyze aggression level
        aggressive_actions = sum(1 for a in actions 
                               if a.get("action", "") in ["bet", "raise", "BET", "RAISE"])
        
        if aggressive_actions >= 2:
            lines.append("Analysis: Heavy aggression this round")
        elif aggressive_actions == 1:
            lines.append("Analysis: Some aggression this round")
        else:
            lines.append("Analysis: Passive action this round")
        
        return "\n".join(lines)

    def get_opponent_status_summary(self, opponents: List[Dict[str, Any]]) -> str:
        """Get summary of opponents' status"""
        
        if not opponents:
            return "No opponents remaining"
        
        lines = []
        lines.append(f"Opponents remaining: {len(opponents)}")
        
        for opp in opponents:
            name = opp.get("name", "Unknown")
            chips = opp.get("chips", 0)
            current_bet = opp.get("current_bet", 0)
            is_all_in = opp.get("is_all_in", False)
            position = opp.get("position", "unknown")
            
            status_parts = [f"{chips} chips"]
            if current_bet > 0:
                status_parts.append(f"bet {current_bet}")
            if is_all_in:
                status_parts.append("ALL-IN")
            
            status = ", ".join(status_parts)
            lines.append(f"  {name} ({position}): {status}")
        
        # Add stack analysis
        total_chips = sum(opp.get("chips", 0) for opp in opponents)
        avg_stack = total_chips / len(opponents) if opponents else 0
        
        lines.append(f"Average opponent stack: {avg_stack:.0f} chips")
        
        return "\n".join(lines)

    def get_betting_round_summary(self, betting_round: str, pot: int, 
                                 current_bet: int, actions_count: int = 0) -> str:
        """Get summary of current betting round"""
        
        lines = []
        lines.append(f"=== {betting_round} BETTING ===")
        lines.append(f"Pot: {pot} chips")
        lines.append(f"Current bet to call: {current_bet} chips")
        lines.append(f"Actions taken this round: {actions_count}")
        
        # Round-specific guidance
        if betting_round == "PREFLOP":
            lines.append("Focus: Hand selection, position, opponent tendencies")
        elif betting_round == "FLOP":
            lines.append("Focus: Board texture, drawing potential, continuation betting")
        elif betting_round == "TURN":
            lines.append("Focus: Hand strength, pot commitment, river planning")
        elif betting_round == "RIVER":
            lines.append("Focus: Value betting, bluff catching, showdown value")
        
        return "\n".join(lines)

    def check_blind_defense_situation(self, player_current_bet: int, 
                                     betting_round: str, pot: int,
                                     amount_to_call: int) -> str:
        """Check if this is a blind defense situation"""
        
        if betting_round != "PREFLOP" or player_current_bet <= 0:
            return "Not a blind defense situation"
        
        lines = []
        lines.append("BLIND DEFENSE SITUATION DETECTED")
        lines.append(f"You posted {player_current_bet} chips as a blind")
        
        if amount_to_call > 0:
            pot_odds_ratio = pot / amount_to_call
            lines.append(f"Pot odds for defense: {pot_odds_ratio:.1f}:1")
            
            # Blind defense guidance
            if pot_odds_ratio >= 3.0:
                lines.append("STRONG DEFENSE RECOMMENDED:")
                lines.append("- Fold only the weakest 15-20% of hands")
                lines.append("- Examples to fold: 72o, 83o, 92o, 84o")
            elif pot_odds_ratio >= 2.0:
                lines.append("MODERATE DEFENSE RECOMMENDED:")
                lines.append("- Defend with ~60-70% of hands")
                lines.append("- Fold weak offsuit hands and bottom pairs")
            else:
                lines.append("TIGHT DEFENSE RECOMMENDED:")
                lines.append("- Defend with stronger hands only")
                lines.append("- Consider folding marginal hands despite investment")
            
            lines.append("Key principle: Your blind investment is sunk cost")
            lines.append("Decision should be based on future profitability")
        else:
            lines.append("No additional call required for blind defense")
        
        return "\n".join(lines)

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
    server = GameStateServer()
    server.run()


if __name__ == "__main__":
    main()