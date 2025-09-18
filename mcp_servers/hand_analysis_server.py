




#!/usr/bin/env python3
"""
hand_analysis_server.py - Version with file logging for production debugging

This version logs to a file so we can see what happens during actual poker games
"""

import json
import sys
import os
import traceback
import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import signal
import time
import threading
from contextlib import contextmanager

import sys, os, time

# Ensure immediate flush to parent process / console
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def debug_log(msg: str) -> None:
    """Safe per-process logger (stderr only, avoids JSON protocol pollution)."""
    pid = os.getpid()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [HAND_SERVER {pid}] {msg}", file=sys.stderr, flush=True)





# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import src.feature_engineering as fe
    from src.agent_utils import card_to_str
    from src.core_poker_mechanics import Card, Suit
    from src.logging_config import logger
    #debug_log("Successfully imported poker modules")
except ImportError as e:
    #debug_log(f"Import error: {e}")
    sys.exit(1)



"""
Numba compilation for src/feature_engineering.py
"""

import numpy as np
import sys
import os

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

### Compile Numba functions
def warmup_numba():
    """
    Compile and cache all numba-specialized variants we rely on,
    with representative inputs that cover major branches.
    """
    t0 = time.perf_counter()

    # Representative hole cards
    hole_cases = [
        (np.array([14, 13], dtype=np.int32), np.array([0, 1], dtype=np.int32)),  # offsuit AK
        (np.array([10, 8], dtype=np.int32),  np.array([3, 3], dtype=np.int32)),  # suited T8
        (np.array([9,  9], dtype=np.int32),  np.array([2, 2], dtype=np.int32)),  # pair 99
        (np.array([8, 7], dtype=np.int32),   np.array([1, 1], dtype=np.int32)),  # suited connectors
        (np.array([5, 4], dtype=np.int32),   np.array([2, 3], dtype=np.int32)),  # wheel draw potential
    ]

    # Representative boards
    boards = [
        (np.array([], dtype=np.int32), np.array([], dtype=np.int32)),                              # preflop
        (np.array([2, 7, 12], dtype=np.int32), np.array([0, 1, 2], dtype=np.int32)),              # dry flop
        (np.array([6, 9, 13], dtype=np.int32), np.array([0, 1, 2], dtype=np.int32)),              # open-ended
        (np.array([5, 9, 13], dtype=np.int32), np.array([0, 1, 2], dtype=np.int32)),              # gutshot
        (np.array([14, 2, 3], dtype=np.int32), np.array([0, 1, 2], dtype=np.int32)),              # wheel draw
        (np.array([2, 9, 13], dtype=np.int32), np.array([3, 3, 1], dtype=np.int32)),              # flush draw
        (np.array([9, 7, 13], dtype=np.int32), np.array([3, 3, 3], dtype=np.int32)),              # made flush
        (np.array([2, 7, 12, 5], dtype=np.int32), np.array([0, 1, 2, 3], dtype=np.int32)),        # turn
        (np.array([2, 7, 12, 5, 9], dtype=np.int32), np.array([0, 1, 2, 3, 0], dtype=np.int32)),  # river
    ]

    opponent_counts = [1, 2, 6, 9]  # edge values are enough

    for hole_ranks, hole_suits in hole_cases:
        for board_ranks, board_suits in boards:
            fe.extract_hand_strength_scalar(hole_ranks, board_ranks, hole_suits, board_suits)
            if board_ranks.size >= 3:
                fe.evaluate_best_hand(hole_ranks, board_ranks, hole_suits, board_suits)

            fe.calculate_straight_probability(hole_ranks, hole_suits, board_ranks, board_suits)
            fe.calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)

            for n_opponents in opponent_counts:
                fe.calculate_opponent_straight_probability(
                    hole_ranks, hole_suits, board_ranks, board_suits, n_opponents
                )
                fe.calculate_opponent_flush_probability(
                    hole_ranks, hole_suits, board_ranks, board_suits, n_opponents
                )

    t1 = time.perf_counter()
    logger.info(f"Numba cache warmup complete in {t1 - t0:.3f}s")



# Run warmup once at startup
logger.info('Numba: starting compilation (in hand_analysis_server.py)')
warmup_numba()
logger.info('Numba: finished compilation (in hand_analysis_server.py)')






class TimeoutError(Exception):
    pass

@contextmanager
def timeout_after(seconds, function_name="unknown"):
    """Windows-compatible timeout using threading"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            yield
        except Exception as e:
            exception[0] = e
    
    def run_with_timeout():
        try:
            # This is a bit tricky - we need to run the actual code in the context
            # For now, we'll use a simpler approach with just time tracking
            pass
        except Exception as e:
            exception[0] = e
    
    start_time = time.time()
    yield  # Let the code run
    elapsed = time.time() - start_time
    
    if elapsed > seconds:
        raise TimeoutError(f"Function '{function_name}' took {elapsed:.1f}s (limit: {seconds}s)")
    
    if exception[0]:
        raise exception[0]

class HandAnalysisServer:
    """Hand analysis server with comprehensive logging"""
    
    def __init__(self):
        #debug_log(f"Server initializing... PID={os.getpid()}")
        self.request_count = 0
        self.start_time = datetime.datetime.now()
        
        # Clear the log file on startup
        #try:
        #    with open(LOG_FILE, "w") as f:
        #        f.write(f"Hand Analysis Server started at {self.start_time}\n")
        #        f.write(f"PID: {os.getpid()}\n")
        #        f.write(f"Python: {sys.executable}\n")
        #        f.write(f"Working directory: {os.getcwd()}\n")
        #        f.write("=" * 50 + "\n")
        #except:
        #    pass
        
        self.tools = {
            "get_hand_features": {
                "description": "Get comprehensive hand analysis including strength, draws, and probabilities",
                "method": self.get_hand_features
            },
            #"get_hand_strength": {
            #    "description": "Get normalized hand strength score (0.0 to 1.0)",
            #    "method": self.get_hand_strength
            #},
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
        #debug_log(f"Server initialized with {len(self.tools)} tools")
    
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
        """Get comprehensive hand analysis with timeout protection and detailed logging"""
        #debug_log(f"=== HAND ANALYSIS REQUEST ===")
        #debug_log(f"Hole cards: {hole_cards}")
        #debug_log(f"Board cards: {board_cards}")
        #debug_log(f"Num opponents: {num_opponents}")
        
        start_time = time.time()
        
        try:
            #debug_log("Parsing cards...")
            hole_ranks, hole_suits = self.parse_cards(hole_cards)
            board_ranks, board_suits = self.parse_cards(board_cards)
            
            hole_ranks = hole_ranks.astype(np.int32, copy=False)
            board_ranks = board_ranks.astype(np.int32, copy=False)
            hole_suits = hole_suits.astype(np.int32, copy=False)
            board_suits = board_suits.astype(np.int32, copy=False)

            #debug_log(f"Cards parsed in {time.time() - start_time:.3f}s")
            
            features = []
            
            # Hand strength (if board exists)
            if len(board_cards) > 0:
                # Test each function individually with timing
                
                #debug_log("Computing hand strength...")
                func_start = time.time()
                try:
                    strength = fe.extract_hand_strength_scalar(hole_ranks, board_ranks, hole_suits, board_suits)
                    #debug_log(f"Hand strength computed in {time.time() - func_start:.3f}s")
                    features.append(f"Hand strength: {strength:.3f} (0.0-1.0 scale)")
                except Exception as e:
                    #debug_log(f"Hand strength failed after {time.time() - func_start:.3f}s: {e}")
                    features.append("Hand strength: Error")
                    
                #debug_log("Computing current hand name...")
                func_start = time.time()
                try:
                    hand_name = fe.best_hand_string(hole_ranks, board_ranks, hole_suits, board_suits)
                    #debug_log(f"Hand name computed in {time.time() - func_start:.3f}s")
                    features.append(f"Current hand: {hand_name}")
                except Exception as e:
                    #debug_log(f"Hand name failed after {time.time() - func_start:.3f}s: {e}")
                    features.append("Current hand: Error")
                    
                #debug_log("Computing straight probabilities...")
                func_start = time.time()
                try:
                    straight_prob = fe.calculate_straight_probability(hole_ranks, hole_suits, board_ranks, board_suits)
                    #debug_log(f"Player straight probability computed in {time.time() - func_start:.3f}s")
                    
                    # This is likely the problematic function
                    opponent_start = time.time()
                    opponent_straight_prob = fe.calculate_opponent_straight_probability(
                        hole_ranks, hole_suits, board_ranks, board_suits, num_opponents
                    )
                    #debug_log(f"Opponent straight probability computed in {time.time() - opponent_start:.3f}s")
                    
                    features.append(f"Straight probability: {straight_prob:.1%} (yours) vs {opponent_straight_prob:.1%} (opponents)")
                except Exception as e:
                    #debug_log(f"Straight probabilities failed after {time.time() - func_start:.3f}s: {e}")
                    features.append("Straight probability: Error")
                    
                #debug_log("Computing flush probabilities...")
                func_start = time.time()
                try:
                    flush_prob = fe.calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)
                    #debug_log(f"Player flush probability computed in {time.time() - func_start:.3f}s")
                    
                    # This might also be problematic
                    opponent_flush_start = time.time()
                    opponent_flush_prob = fe.calculate_opponent_flush_probability(
                        hole_ranks, hole_suits, board_ranks, board_suits, num_opponents
                    )
                    #debug_log(f"Opponent flush probability computed in {time.time() - opponent_flush_start:.3f}s")
                    
                    features.append(f"Flush probability: {flush_prob:.1%} (yours) vs {opponent_flush_prob:.1%} (opponents)")
                except Exception as e:
                    #debug_log(f"Flush probabilities failed after {time.time() - func_start:.3f}s: {e}")
                    features.append("Flush probability: Error")
                    
                #debug_log("Computing board analysis...")
                func_start = time.time()
                try:
                    board_pattern = fe.best_board_pattern(board_ranks, board_suits)
                    #debug_log(f"Board analysis computed in {time.time() - func_start:.3f}s")
                    if board_pattern:
                        features.append(f"Board shows: {board_pattern} (all players have at least this)")
                except Exception as e:
                    print(f"Board analysis failed after {time.time() - func_start:.3f}s: {e}")
            
            # Preflop analysis (existing code - this doesn't timeout)
            if len(board_cards) == 0:
                #debug_log("Computing preflop analysis...")
                func_start = time.time()
                
                if len(hole_cards) >= 2:
                    rank1, rank2 = hole_ranks[0], hole_ranks[1]
                    suited = hole_suits[0] == hole_suits[1]
                    
                    # Sort ranks for consistent analysis (high to low)
                    high_rank, low_rank = (rank1, rank2) if rank1 >= rank2 else (rank2, rank1)
                    
                    rank_names = {14: "Ace", 13: "King", 12: "Queen", 11: "Jack", 10: "Ten"}
                    
                    # Pocket pair detection
                    if rank1 == rank2:
                        ranking = fe.get_preflop_ranking(rank1, rank1, False)
                        features.append(f"Pocket pair: {rank1}s")
                    else:
                        # Non-pair hand analysis
                        high_card_name = rank_names.get(high_rank, str(high_rank))
                        low_card_name = rank_names.get(low_rank, str(low_rank))
                        suit_text = "Suited" if suited else "Unsuited"
                        
                        features.append(f"{suit_text} {high_card_name.lower()} high, {low_card_name.lower()} kicker")
                        
                        # Gap analysis
                        gap = high_rank - low_rank - 1
                        if gap == 0:
                            features.append("Connected (no gap)")
                        else:
                            features.append(f"{gap}-rank gap")
                        
                        # Special holdings
                        if high_rank >= 10 and low_rank >= 10:
                            features.append("Broadway cards")
                        
                        if suited and gap <= 1:
                            features.append("Suited connector")
                        
                        # Hand ranking
                        ranking = fe.get_preflop_ranking(high_rank, low_rank, suited)
                        suit_desc = "suited" if suited else "offsuit"
                        hand_desc = f"{high_card_name.lower()}-{low_card_name.lower()} {suit_desc}"
                    
                    features.append(f"Of 169 possible hands, {hand_desc if 'hand_desc' in locals() else 'pocket ' + str(rank1) + 's'} ranks {fe.format_ordinal(ranking)} strongest")
                
                #debug_log(f"Preflop analysis computed in {time.time() - func_start:.3f}s")
            
            total_time = time.time() - start_time
            result = "\n".join(features)
            #debug_log(f"=== ANALYSIS COMPLETE ===")
            #debug_log(f"Total time: {total_time:.3f}s")
            #debug_log(f"Result length: {len(result)} chars")
            #debug_log(f"Features computed: {len(features)}")
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            #debug_log(f"GENERAL ERROR after {total_time:.3f}s: {e}")
            #debug_log(f"Failed on cards: hole={hole_cards}, board={board_cards}")
            return f"Hand analysis error after {total_time:.3f}s: {str(e)}"

    def get_hand_strength(self, hole_cards: List[str], board_cards: List[str]) -> str:
        """Get normalized hand strength"""
        #debug_log(f"get_hand_strength called: hole={hole_cards}, board={board_cards}")
        try:
            hole_ranks, hole_suits = self.parse_cards(hole_cards)
            board_ranks, board_suits = self.parse_cards(board_cards)
            
            strength = fe.extract_hand_strength_scalar(hole_ranks, board_ranks, hole_suits, board_suits)
            result = f"{strength:.3f}"
            #debug_log(f"get_hand_strength completed: {result}")
            return result
        except Exception as e:
            #debug_log(f"get_hand_strength error: {e}")
            raise

    def get_current_hand(self, hole_cards: List[str], board_cards: List[str]) -> str:
        """Get current best hand name"""
        #debug_log(f"get_current_hand called: hole={hole_cards}, board={board_cards}")
        try:
            hole_ranks, hole_suits = self.parse_cards(hole_cards)
            board_ranks, board_suits = self.parse_cards(board_cards)
            
            result = fe.best_hand_string(hole_ranks, board_ranks, hole_suits, board_suits)
            #debug_log(f"get_current_hand completed: {result}")
            return result
        except Exception as e:
            #debug_log(f"get_current_hand error: {e}")
            raise

    def get_straight_probabilities(self, hole_cards: List[str], board_cards: List[str], num_opponents: int) -> str:
        """Get straight probabilities for player and opponents"""
        #debug_log(f"get_straight_probabilities called")
        try:
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
            
            json_result = json.dumps(result)
            #debug_log(f"get_straight_probabilities completed")
            return json_result
        except Exception as e:
            #ebug_log(f"get_straight_probabilities error: {e}")
            raise

    def get_flush_probabilities(self, hole_cards: List[str], board_cards: List[str], num_opponents: int) -> str:
        """Get flush probabilities for player and opponents"""
        #debug_log(f"get_flush_probabilities called")
        try:
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
            
            json_result = json.dumps(result)
            #debug_log(f"get_flush_probabilities completed")
            return json_result
        except Exception as e:
            #debug_log(f"get_flush_probabilities error: {e}")
            raise

    def get_board_analysis(self, board_cards: List[str]) -> str:
        """Analyze board texture"""
        #debug_log(f"get_board_analysis called: board={board_cards}")
        try:
            if len(board_cards) < 3:
                return "Need at least 3 board cards for analysis"
            
            board_ranks, board_suits = self.parse_cards(board_cards)
            board_pattern = fe.best_board_pattern(board_ranks, board_suits)
            
            if board_pattern:
                result = f"Board texture: {board_pattern} (all players have at least this hand)"
            else:
                result = "Board texture: High card only (no made hands on board)"
            
            #debug_log(f"get_board_analysis completed")
            return result
        except Exception as e:
            #debug_log(f"get_board_analysis error: {e}")
            raise

    def check_suited_hole_cards(self, hole_cards: List[str]) -> str:
        """Check if hole cards are suited"""
        #debug_log(f"check_suited_hole_cards called: hole={hole_cards}")
        try:
            if len(hole_cards) != 2:
                return json.dumps({"error": "Need exactly 2 hole cards"})
            
            _, hole_suits = self.parse_cards(hole_cards)
            suited = hole_suits[0] == hole_suits[1]
            
            result = {
                "suited": suited,
                "description": "Hole cards are suited" if suited else "Hole cards are unsuited"
            }
            
            json_result = json.dumps(result)
            #debug_log(f"check_suited_hole_cards completed")
            return json_result
        except Exception as e:
            #debug_log(f"check_suited_hole_cards error: {e}")
            raise

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request"""
        self.request_count += 1
        uptime = datetime.datetime.now() - self.start_time
        #debug_log(f"Handling request #{self.request_count} (uptime: {uptime}): {request.get('method', 'unknown')}")
        
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
                
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": tools
                    }
                }
                #debug_log(f"tools/list response prepared")
                return response
            
            elif method == "tools/call":
                # Call a specific tool
                params = request.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                #debug_log(f"tools/call: {tool_name} with args {arguments}")
                
                if tool_name not in self.tools:
                    debug_log(f"Tool not found: {tool_name}")
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
                
                #debug_log(f"About to call method: {tool_name}")
                #debug_log(f"Arguments: {arguments}")
                #debug_log(f"Method object: {tool_method}")
                
                result_text = tool_method(**arguments)
                
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"text": result_text}]
                    }
                }
                #debug_log(f"tools/call completed successfully for {tool_name}")
                return response
            
            else:
                debug_log(f"Unknown method: {method}")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
        
        except Exception as e:
            #debug_log(f"Request handling error: {e}")
            #debug_log(f"Traceback: {traceback.format_exc()}")
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
        #debug_log("Server starting main loop...")
        
        try:
            while True:
                # Read a line from stdin
                #debug_log("Waiting for input...")
                line = sys.stdin.readline()
                
                if not line:
                    #debug_log("EOF received on stdin, server exiting")
                    break
                
                line = line.strip()
                if not line:
                    #debug_log("Empty line received, continuing...")
                    continue
                
                #debug_log(f"Received input: {line[:100]}{'...' if len(line) > 100 else ''}")
                
                try:
                    # Parse JSON request
                    request = json.loads(line)
                    
                    # Handle the request
                    response = self.handle_request(request)
                    
                    # Send JSON response
                    response_json = json.dumps(response)
                    print(response_json)
                    sys.stdout.flush()
                    #debug_log(f"Response sent: {response_json[:100]}{'...' if len(response_json) > 100 else ''}")
                    
                except json.JSONDecodeError as e:
                    #debug_log(f"JSON decode error: {e}")
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
            print("Server interrupted by keyboard")
        except Exception as e:
            print(f"Server error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            uptime = datetime.datetime.now() - self.start_time
            #debug_log(f"Server shutting down after {uptime}, processed {self.request_count} requests")


def main():
    """Main entry point"""
    #debug_log("Hand Analysis Server starting...")
    server = HandAnalysisServer()
    server.run()
    #debug_log("Hand Analysis Server terminated")


if __name__ == "__main__":
    main()