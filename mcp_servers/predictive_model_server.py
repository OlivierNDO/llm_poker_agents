"""
predictive_model_server.py

Simple, reliable server that just works
"""
import os
import sys
import json
import re
from typing import Dict, List, Any

# Ensure proper buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.predictive_model import PokerPredictor
    import src.feature_engineering as fe
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

CARD_RE = re.compile(r'^(10|[2-9TJQKA])[cdhs]$')

def validate_inputs(hole_cards: List[str], board_cards: List[str], num_players: int) -> None:
    """Validate inputs"""
    if not isinstance(hole_cards, list) or not isinstance(board_cards, list):
        raise ValueError('hole_cards and board_cards must be lists')

    if len(hole_cards) != 2:
        raise ValueError(f'need exactly 2 hole cards, got {len(hole_cards)}')

    if len(board_cards) > 5:
        raise ValueError(f'board cannot have more than 5 cards, got {len(board_cards)}')

    if not isinstance(num_players, int) or num_players < 2 or num_players > 9:
        raise ValueError(f'num_players must be 2-9, got {num_players}')

    # Check card format
    for c in hole_cards + board_cards:
        if not isinstance(c, str):
            raise ValueError(f'cards must be strings, got {type(c)}: {c}')
        if not CARD_RE.match(c):
            raise ValueError(f'invalid card: "{c}"')

    # Check for duplicates
    all_cards = hole_cards + board_cards
    if len(set(all_cards)) != len(all_cards):
        raise ValueError(f'duplicate cards: {all_cards}')

class SimplePredictiveServer:
    """Simple server for Monte Carlo predictions"""
    
    def __init__(self):
        self.predictor = self._create_predictor()
        self.tools = {
            "predict_outcome_monte_carlo": self.predict_monte_carlo,
            "get_win_probability": self.get_win_probability
        }

    def _create_predictor(self):
        """Create a predictor instance for Monte Carlo only"""
        try:
            # Create instance without loading ML model
            predictor = PokerPredictor.__new__(PokerPredictor)
            predictor.is_three_class = True
            return predictor
        except Exception as e:
            print(f"Predictor creation failed: {e}", file=sys.stderr, flush=True)
            return None

    def predict_monte_carlo(self, hole_cards: List[str], board_cards: List[str],
                           num_players: int, iterations: int = 2500) -> str:
        """Monte Carlo prediction"""
        try:
            validate_inputs(hole_cards, board_cards, num_players)
            
            # Use 10k iterations always
            iterations = 2500
            
            if self.predictor:
                result = self.predictor.monte_carlo(hole_cards, board_cards, num_players, iterations)
            else:
                # Fallback: create temporary predictor
                from src.predictive_model import PokerPredictor
                temp_predictor = PokerPredictor.__new__(PokerPredictor)
                result = temp_predictor.monte_carlo(hole_cards, board_cards, num_players, iterations)
            
            formatted = {
                "method": "Monte_Carlo",
                "win_probability": round(result["win_probability"], 4),
                "tie_probability": round(result["tie_probability"], 4),
                "loss_probability": round(result["loss_probability"], 4),
                "win_percentage": f"{result['win_probability']:.1%}",
                "tie_percentage": f"{result['tie_probability']:.1%}",
                "loss_percentage": f"{result['loss_probability']:.1%}",
                "samples": result["samples"],
                "iterations": iterations
            }
            
            return json.dumps(formatted)
            
        except Exception as e:
            return json.dumps({"error": f"Monte Carlo error: {str(e)}"})

    def get_win_probability(self, hole_cards: List[str], board_cards: List[str],
                           num_players: int) -> str:
        """Quick win probability"""
        try:
            validate_inputs(hole_cards, board_cards, num_players)
            
            # Use 1k iterations for speed
            iterations = 1000
            
            if self.predictor:
                result = self.predictor.monte_carlo(hole_cards, board_cards, num_players, iterations)
            else:
                from src.predictive_model import PokerPredictor
                temp_predictor = PokerPredictor.__new__(PokerPredictor)
                result = temp_predictor.monte_carlo(hole_cards, board_cards, num_players, iterations)
            
            return f"{result['win_probability']:.3f}"
                
        except Exception as e:
            return f"Error: {str(e)}"

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request"""
        try:
            method = request.get("method")
            request_id = request.get("id")
            
            if method == "tools/list":
                tools = [
                    {
                        "name": "predict_outcome_monte_carlo",
                        "description": "Monte Carlo simulation (2,500 iterations)"
                    },
                    {
                        "name": "get_win_probability", 
                        "description": "Quick win probability (1,000 iterations)"
                    }
                ]
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": tools}
                }
            
            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if tool_name not in self.tools:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        }
                    }
                
                # Call the tool
                tool_method = self.tools[tool_name]
                result_text = tool_method(**arguments)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"text": result_text}]}
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}"
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
        """Run the server"""
        print("Starting simple predictive server", file=sys.stderr, flush=True)
        
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    response = self.handle_request(request)
                    response_json = json.dumps(response)
                    print(response_json, flush=True)
                    
                except json.JSONDecodeError as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {str(e)}"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                    
        except KeyboardInterrupt:
            print("Server interrupted", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    server = SimplePredictiveServer()
    server.run()