#!/usr/bin/env python3
"""
hand_analysis_server.py

Simplified JSON-RPC server for poker hand analysis.
Provides hand analysis tools without the full MCP framework complexity.
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional
import numpy as np

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import src.feature_engineering as fe
    from src.agent_utils import card_to_str
    from src.core_poker_mechanics import Card, Suit
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


class HandAnalysisServer:
    """Simplified hand analysis server using JSON-RPC over stdio"""
    
    def __init__(self):
        self.tools = {
            "get_hand_features": {
                "description": "Get comprehensive hand analysis including strength, draws, and probabilities",
                "method": self.get_hand_features
            },
            "get_hand_strength": {
                "description": "Get normalized hand strength score (0.0 to 1.0)",
                "method": self.get_hand_strength
            },
            "get_current_hand": {
                "description": "Get the name of the current best hand",
                "method": self.get_current_hand
            },
            "get_straight_probabilities": {
                "description": "Get straight probability for player and opponents",
                "method": self.get_straight_probabilities
            },
            "get_flush_probabilities": {
                "description": "Get flush probability for player and opponents", 
                "method": self.get_flush_probabilities
            },
            "get_board_analysis": {
                "description": "Analyze the board texture",
                "method": self.get_board_analysis
            },
            "check_suited_hole_cards": {
                "description": "Check if hole cards are suited",
                "method": self.check_suited_hole_cards
            }
        }
    
    def parse_card(self, card_str: str) -> tuple[int, int]:
        """Convert string like 'Ad' to (rank, suit)"""
        rank_str_to_int = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
            'K': 13, 'A': 14
        }
        suit_str_to_int = {'c': 0, 'h': 1, 'd': 2, 's': 3}
        
        if len(card_str) != 2:
            raise ValueError(f"Invalid card format: {card_str}")
            
        rank = rank_str_to_int.get(card_str[0].upper())
        suit = suit_str_to_int.get(card_str[1].lower())
        
        if rank is None or suit is None:
            raise ValueError(f"Invalid card: {card_str}")
            
        return rank, suit

    def parse_cards(self, card_strings: List[str]) -> tuple[np.ndarray, np.ndarray]:
        """Parse list of card strings into rank and suit arrays"""
        if not card_strings:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            
        parsed = [self.parse_card(card) for card in card_strings]
        ranks = np.array([r for r, s in parsed], dtype=np.int32)
        suits = np.array([s for r, s in parsed], dtype=np.int32)
        return ranks, suits

    def get_hand_features(self, hole_cards: List[str], board_cards: List[str], num_opponents: int) -> str:
        """Get comprehensive hand analysis"""
        hole_ranks, hole_suits = self.parse_cards(hole_cards)
        board_ranks, board_suits = self.parse_cards(board_cards)
        
        features = []
        
        # Hand strength (if board exists)
        if len(board_cards) > 0:
            strength = fe.extract_hand_strength_scalar(hole_ranks, board_ranks, hole_suits, board_suits)
            features.append(f"Hand strength: {strength:.3f} (0.0-1.0 scale)")
            
            # Current hand name
            hand_name = fe.best_hand_string(hole_ranks, board_ranks, hole_suits, board_suits)
            features.append(f"Current hand: {hand_name}")
            
            # Straight probabilities
            straight_prob = fe.calculate_straight_probability(hole_ranks, hole_suits, board_ranks, board_suits)
            opponent_straight_prob = fe.calculate_opponent_straight_probability(
                hole_ranks, hole_suits, board_ranks, board_suits, num_opponents
            )
            features.append(f"Straight probability: {straight_prob:.1%} (yours) vs {opponent_straight_prob:.1%} (opponents)")
            
            # Flush probabilities  
            flush_prob = fe.calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)
            opponent_flush_prob = fe.calculate_opponent_flush_probability(
                hole_ranks, hole_suits, board_ranks, board_suits, num_opponents
            )
            features.append(f"Flush probability: {flush_prob:.1%} (yours) vs {opponent_flush_prob:.1%} (opponents)")
            
            # Board analysis
            board_pattern = fe.best_board_pattern(board_ranks, board_suits)
            if board_pattern:
                features.append(f"Board shows: {board_pattern} (all players have at least this)")
        
        # Preflop analysis
        if len(board_cards) == 0:
            suited = hole_suits[0] == hole_suits[1] if len(hole_suits) >= 2 else False
            features.append(f"Hole cards are {'suited' if suited else 'unsuited'}")
        
        return "\n".join(features)

    def get_hand_strength(self, hole_cards: List[str], board_cards: List[str]) -> str:
        """Get normalized hand strength"""
        hole_ranks, hole_suits = self.parse_cards(hole_cards)
        board_ranks, board_suits = self.parse_cards(board_cards)
        
        strength = fe.extract_hand_strength_scalar(hole_ranks, board_ranks, hole_suits, board_suits)
        return f"{strength:.3f}"

    def get_current_hand(self, hole_cards: List[str], board_cards: List[str]) -> str:
        """Get current best hand name"""
        hole_ranks, hole_suits = self.parse_cards(hole_cards)
        board_ranks, board_suits = self.parse_cards(board_cards)
        
        return fe.best_hand_string(hole_ranks, board_ranks, hole_suits, board_suits)

    def get_straight_probabilities(self, hole_cards: List[str], board_cards: List[str], num_opponents: int) -> str:
        """Get straight probabilities for player and opponents"""
        hole_ranks, hole_suits = self.parse_cards(hole_cards)
        board_ranks, board_suits = self.parse_cards(board_cards)
        
        player_prob = fe.calculate_straight_probability(hole_ranks, hole_suits, board_ranks, board_suits)
        opponent_prob = fe.calculate_opponent_straight_probability(
            hole_ranks, hole_suits, board_ranks, board_suits, num_opponents
        )
        
        result = {
            "player_probability": round(player_prob, 4),
            "opponent_probability": round(opponent_prob, 4),
            "player_percentage": f"{player_prob:.1%}",
            "opponent_percentage": f"{opponent_prob:.1%}"
        }
        
        return json.dumps(result)

    def get_flush_probabilities(self, hole_cards: List[str], board_cards: List[str], num_opponents: int) -> str:
        """Get flush probabilities for player and opponents"""
        hole_ranks, hole_suits = self.parse_cards(hole_cards)
        board_ranks, board_suits = self.parse_cards(board_cards)
        
        player_prob = fe.calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)
        opponent_prob = fe.calculate_opponent_flush_probability(
            hole_ranks, hole_suits, board_ranks, board_suits, num_opponents
        )
        
        result = {
            "player_probability": round(player_prob, 4),
            "opponent_probability": round(opponent_prob, 4),
            "player_percentage": f"{player_prob:.1%}",
            "opponent_percentage": f"{opponent_prob:.1%}"
        }
        
        return json.dumps(result)

    def get_board_analysis(self, board_cards: List[str]) -> str:
        """Analyze board texture"""
        if len(board_cards) < 3:
            return "Need at least 3 board cards for analysis"
        
        board_ranks, board_suits = self.parse_cards(board_cards)
        board_pattern = fe.best_board_pattern(board_ranks, board_suits)
        
        if board_pattern:
            return f"Board texture: {board_pattern} (all players have at least this hand)"
        else:
            return "Board texture: High card only (no made hands on board)"

    def check_suited_hole_cards(self, hole_cards: List[str]) -> str:
        """Check if hole cards are suited"""
        if len(hole_cards) != 2:
            return json.dumps({"error": "Need exactly 2 hole cards"})
        
        _, hole_suits = self.parse_cards(hole_cards)
        suited = hole_suits[0] == hole_suits[1]
        
        result = {
            "suited": suited,
            "description": "Hole cards are suited" if suited else "Hole cards are unsuited"
        }
        
        return json.dumps(result)

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
    server = HandAnalysisServer()
    server.run()


if __name__ == "__main__":
    main()