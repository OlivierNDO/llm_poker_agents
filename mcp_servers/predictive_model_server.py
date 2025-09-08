"""
predictive_model_server.py

Fixed server with TensorFlow cleanup to prevent hanging.
"""
import os
import sys

# TensorFlow environment variables MUST be set before importing TF
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import asyncio
import json
import re
import gc
from typing import Dict, List, Any, Optional

# Ensure proper stdout buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.predictive_model import PokerPredictor
    from src.logging_config import logger
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

CARD_RE = re.compile(r'^(10|[2-9TJQKA])[cdhs]$')

def _validate_inputs(
    hole_cards: List[str],
    board_cards: List[str],
    num_players: int
) -> None:
    """Validate inputs and raise clear errors if invalid."""
    print(f'[PREDICTIVE_SERVER] Validating inputs:', file=sys.stderr, flush=True)
    print(f'  hole_cards={hole_cards}', file=sys.stderr, flush=True)
    print(f'  board_cards={board_cards}', file=sys.stderr, flush=True)
    print(f'  num_players={num_players}', file=sys.stderr, flush=True)

    if not isinstance(hole_cards, list) or not isinstance(board_cards, list):
        raise ValueError('hole_cards and board_cards must be lists of strings.')

    if len(hole_cards) != 2:
        raise ValueError(f'exactly 2 hole cards required, got {len(hole_cards)}')

    if len(board_cards) > 5:
        raise ValueError(f'board cannot have more than 5 cards, got {len(board_cards)}')

    if not isinstance(num_players, int) or num_players < 2 or num_players > 9:
        raise ValueError(f'num_players must be int in [2,9], got {num_players}')

    # Format check
    for c in hole_cards + board_cards:
        if not isinstance(c, str):
            raise ValueError(f'cards must be strings like "As", got {type(c)}: {c}')
        if not CARD_RE.match(c):
            raise ValueError(f'invalid card string: "{c}"')

    # Duplicate check
    all_cards = hole_cards + board_cards
    if len(set(all_cards)) != len(all_cards):
        raise ValueError(f'duplicate cards in inputs: {all_cards}')

class PredictiveModelServer:
    """JSON-RPC server for poker outcome predictions with TensorFlow cleanup"""
    
    def __init__(self):
        print("Initializing PredictiveModelServer...", file=sys.stderr, flush=True)
        self.predictor = None
        self.model_available = False
        self.prediction_count = 0
        
        # Initialize predictor
        self._initialize_predictor()
        
        self.tools = {
            "predict_outcome_ml": {
                "description": "ML PREDICTION: Win/tie/loss probabilities using trained neural network",
                "method": self.predict_outcome_ml
            },
            "predict_outcome_monte_carlo": {
                "description": "Monte Carlo simulation for outcome prediction",
                "method": self.predict_outcome_monte_carlo
            },
            "get_win_probability": {
                "description": "Quick win probability estimate",
                "method": self.get_win_probability
            }
        }

    def _initialize_predictor(self):
        """Initialize the predictor with proper error handling"""
        try:
            self.predictor = PokerPredictor()
            self.model_available = True
            print("Predictive model loaded successfully", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Could not load predictive model: {e}", file=sys.stderr, flush=True)
            self.predictor = None
            self.model_available = False

    def _aggressive_tensorflow_cleanup(self):
        """Aggressive TensorFlow cleanup to prevent memory leaks and hanging"""
        try:
            import tensorflow as tf
            
            # Clear the current session completely
            tf.keras.backend.clear_session()
            
            # Force garbage collection
            gc.collect()
            
            # Try to manually release GPU memory (if using GPU)
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.reset_memory_growth(gpu)
            except:
                pass  # Ignore GPU cleanup errors
            
            # Clear any cached functions
            try:
                tf.compat.v1.reset_default_graph()
            except:
                pass  # This might not be available in TF 2.x
            
            print(f"[PREDICTIVE_SERVER] TensorFlow cleanup completed", file=sys.stderr, flush=True)
            
        except Exception as e:
            print(f"[PREDICTIVE_SERVER] TensorFlow cleanup error: {e}", file=sys.stderr, flush=True)

    def predict_outcome_ml(self, hole_cards: List[str], board_cards: List[str], 
                          num_players: int) -> str:
        """Predict outcome using ML model with aggressive cleanup"""
        print(f"[PREDICTIVE_SERVER] predict_outcome_ml called (call #{self.prediction_count + 1})", file=sys.stderr, flush=True)
        
        try:
            _validate_inputs(hole_cards, board_cards, num_players)
            
            if not self.model_available:
                error_result = json.dumps({"error": "ML model not available"})
                print(f"[PREDICTIVE_SERVER] Model not available", file=sys.stderr, flush=True)
                return error_result
            
            # Call the actual prediction
            print(f"[PREDICTIVE_SERVER] Calling predictor.predict...", file=sys.stderr, flush=True)
            result = self.predictor.predict(hole_cards, board_cards, num_players)
            print(f"[PREDICTIVE_SERVER] Prediction completed", file=sys.stderr, flush=True)
            
            formatted_result = {
                "method": "ML_model",
                "win_probability": round(result["win_probability"], 4),
                "tie_probability": round(result["tie_probability"], 4),
                "loss_probability": round(result["loss_probability"], 4),
                "win_percentage": f"{result['win_probability']:.1%}",
                "tie_percentage": f"{result['tie_probability']:.1%}",
                "loss_percentage": f"{result['loss_probability']:.1%}",
                "predicted_class": result["predicted_class"],
            }
            
            json_result = json.dumps(formatted_result)
            print(f"[PREDICTIVE_SERVER] ML prediction successful", file=sys.stderr, flush=True)
            
            # AGGRESSIVE TENSORFLOW CLEANUP AFTER EACH PREDICTION
            self._aggressive_tensorflow_cleanup()
            
            self.prediction_count += 1
            return json_result
            
        except Exception as e:
            error_msg = f"Error in ML prediction: {str(e)}"
            print(f"[PREDICTIVE_SERVER] {error_msg}", file=sys.stderr, flush=True)
            
            # Even on error, try to cleanup
            self._aggressive_tensorflow_cleanup()
            
            error_result = json.dumps({"error": error_msg})
            return error_result

    def predict_outcome_monte_carlo(self, hole_cards: List[str], board_cards: List[str],
                                   num_players: int, iterations: int = 10000) -> str:
        """Predict outcome using Monte Carlo simulation"""
        try:
            _validate_inputs(hole_cards, board_cards, num_players)
            
            if self.predictor is None:
                from src.predictive_model import PokerPredictor
                temp_predictor = PokerPredictor.__new__(PokerPredictor)
                result = temp_predictor.monte_carlo(hole_cards, board_cards, num_players, iterations)
            else:
                result = self.predictor.monte_carlo(hole_cards, board_cards, num_players, iterations)
            
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
            return json.dumps({"error": f"Error in Monte Carlo prediction: {str(e)}"})

    def get_win_probability(self, hole_cards: List[str], board_cards: List[str],
                           num_players: int) -> str:
        """Get simple win probability"""
        try:
            _validate_inputs(hole_cards, board_cards, num_players)
            
            if self.model_available:
                result = self.predictor.predict(hole_cards, board_cards, num_players)
                # Cleanup after quick prediction too
                self._aggressive_tensorflow_cleanup()
                return f"{result['win_probability']:.3f}"
            else:
                if self.predictor:
                    result = self.predictor.monte_carlo(hole_cards, board_cards, num_players, 1000)
                else:
                    from src.predictive_model import PokerPredictor
                    temp_predictor = PokerPredictor.__new__(PokerPredictor)
                    result = temp_predictor.monte_carlo(hole_cards, board_cards, num_players, 1000)
                return f"{result['win_probability']:.3f}"
                
        except Exception as e:
            return f"Error getting win probability: {str(e)}"

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request"""
        try:
            method = request.get("method")
            request_id = request.get("id")
            
            print(f"[PREDICTIVE_SERVER] Handling request: {method}", file=sys.stderr, flush=True)
            
            if method == "tools/list":
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
                params = request.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                print(f"[PREDICTIVE_SERVER] Calling tool: {tool_name}", file=sys.stderr, flush=True)
                
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
            print(f"[PREDICTIVE_SERVER] Error handling request: {e}", file=sys.stderr, flush=True)
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    def run(self):
        """Run the server with proper I/O handling"""
        print("[PREDICTIVE_SERVER] Starting JSON-RPC server", file=sys.stderr, flush=True)
        
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    print("[PREDICTIVE_SERVER] EOF received, shutting down", file=sys.stderr, flush=True)
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
            print("[PREDICTIVE_SERVER] Interrupted, shutting down", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[PREDICTIVE_SERVER] Server error: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    print("[PREDICTIVE_SERVER] Creating server instance", file=sys.stderr, flush=True)
    server = PredictiveModelServer()
    print("[PREDICTIVE_SERVER] Starting server", file=sys.stderr, flush=True)
    server.run()