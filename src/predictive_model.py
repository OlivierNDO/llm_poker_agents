"""
src/predictive_model.py

predictor = PokerPredictor()
predictor.predict(hole_cards = ['Ad', 'Kd'], board_cards = ['2d', '7d', '9d'], num_players = 6)

"""
import os
import numpy as np
import tensorflow as tf
from typing import List, Dict
import yaml

from src.logging_config import logger
from src.core_poker_mechanics import Card, Suit
from src.model_features import encode_game_state


# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
#config_path = 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
    
    

def parse_card(card_str: str) -> tuple[int, int]:
    rank_str_to_int = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
        '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
        'K': 13, 'A': 14
    }
    suit_str_to_int = {
        'c': 0, 'd': 2, 'h': 1, 's': 3  # Changed: d=2, h=1
    }
    rank = rank_str_to_int[card_str[0].upper()]
    suit = suit_str_to_int[card_str[1].lower()]
    return rank, suit



def create_poker_model_three_class(
    input_size: int,
    hidden_layers: list[int] = None,
    dropout_rate: float = 0.0
) -> tf.keras.Model:
    """
    Create a three-class poker outcome model with residual connections.
    """
    inputs = tf.keras.Input(shape=(input_size,))
    x = inputs
    prev_units = None

    for i, units in enumerate(hidden_layers or []):
        # Project input to first hidden size
        if i == 0:
            x = tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_initializer='he_normal',
                name='input_projection_dense'
            )(x)
            x = tf.keras.layers.BatchNormalization(name='input_projection_bn')(x)
            x = tf.keras.layers.ReLU(name='input_projection_relu')(x)
            prev_units = units
            continue

        shortcut = x

        x = tf.keras.layers.Dense(
            units,
            activation=None,
            kernel_initializer='he_normal',
            name=f'hidden_{i}_dense'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'hidden_{i}_bn')(x)
        x = tf.keras.layers.ReLU(name=f'hidden_{i}_relu')(x)

        x = tf.keras.layers.Dense(
            units,
            activation=None,
            kernel_initializer='he_normal',
            name=f'hidden_{i}_res_dense'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'hidden_{i}_res_bn')(x)

        # If dimensions mismatch, project shortcut
        if prev_units != units:
            shortcut = tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_initializer='he_normal',
                name=f'hidden_{i}_shortcut_proj'
            )(shortcut)

        x = tf.keras.layers.Add(name=f'hidden_{i}_add')([shortcut, x])
        x = tf.keras.layers.ReLU(name=f'hidden_{i}_res_relu')(x)

        if dropout_rate > 0.0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f'hidden_{i}_dropout')(x)

        prev_units = units

    outputs = tf.keras.layers.Dense(
        3,
        activation='softmax',
        name='output'
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


 

class PokerPredictor:
    """
    Predicts poker hand outcomes for player 0 using a trained TensorFlow model,
    or full hand enumeration via eval7.
    """

    def __init__(self,
                 model_path: str = config.get('game').get('model_path'),
                 is_three_class: bool = True,
                 input_size = config.get('game').get('model_input_size'),
                 hidden_layers = config.get('game').get('model_hidden_layers')):
        """
        Initialize the predictor with a model.

        Parameters
        ----------
        model_path : str
            Path to the model weights (.h5) or full model (.keras).
        is_three_class : bool
            Whether the model outputs [loss, tie, win] (vs. just win).
        """
        self.is_three_class = is_three_class
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        
        # If model_path is relative, resolve it relative to repo root
        if not os.path.isabs(model_path):
            # Always resolve relative to repo root (two levels up from this file)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_path = os.path.abspath(os.path.join(repo_root, model_path))
                    
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = create_poker_model_three_class(
            input_size=self.input_size,  # or use dynamic input size if needed
            hidden_layers=self.hidden_layers,
            dropout_rate=0
        )
        
        model.load_weights(model_path)
        print(f"Loaded model weights from {model_path}")
        self.model = model
        
        
    def monte_carlo(
        self,
        hole_cards: List[str],
        board_cards: List[str],
        num_players: int,
        iterations: int = 10_000,
        random_seed: int = 42
    ) -> Dict[str, float]:
        """
        Estimate win/tie/loss probabilities using optimized Monte Carlo simulation.
    
        Parameters
        ----------
        hole_cards : List[str]
            Two hole cards for player 0 (e.g., ['As', 'Kh']).
        board_cards : List[str]  
            0 to 5 board cards (e.g., ['2d', '5h', '9s']).
        num_players : int
            Total number of players in the game.
        iterations : int
            Number of Monte Carlo iterations to run.
        random_seed : Optional[int]
            Seed for reproducibility.
    
        Returns
        -------
        Dict[str, float]
            Dictionary with win, tie, loss probabilities and sample count.
        """
        import random
        from src.core import HandEvaluator
    
        if random_seed is not None:
            random.seed(random_seed)
    
        if len(hole_cards) != 2:
            raise ValueError("Exactly 2 hole cards required for player 0.")
        if len(board_cards) > 5:
            raise ValueError("Board cannot have more than 5 cards.")
        if num_players < 2:
            raise ValueError("At least two players required.")
    
        hero_cards = self._create_cards(hole_cards)
        board = self._create_cards(board_cards)
    
        full_deck = [
            Card(rank, suit)
            for suit in [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
            for rank in range(2, 15)
        ]
        used_cards = set(hero_cards + board)
        available_deck = [card for card in full_deck if card not in used_cards]
    
        cards_needed_for_board = 5 - len(board)
        cards_needed_for_opponents = (num_players - 1) * 2
        total_cards_needed = cards_needed_for_board + cards_needed_for_opponents
    
        if len(available_deck) < total_cards_needed:
            raise ValueError(f"Not enough cards available: need {total_cards_needed}, have {len(available_deck)}")
    
        wins = ties = losses = 0
    
        for _ in range(iterations):
            random.shuffle(available_deck)
    
            complete_board = board + available_deck[:cards_needed_for_board]
    
            opponent_cards = []
            card_idx = cards_needed_for_board
            for _ in range(num_players - 1):
                opp_hole = [available_deck[card_idx], available_deck[card_idx + 1]]
                opponent_cards.append(opp_hole)
                card_idx += 2
    
            hero_seven = hero_cards + complete_board
            hero_rank, hero_tiebreakers = HandEvaluator.evaluate_hand(hero_seven)
            hero_key = (hero_rank.value, hero_tiebreakers)
    
            opponent_keys = [
                HandEvaluator.evaluate_hand(opp + complete_board)
                for opp in opponent_cards
            ]
            opponent_keys = [(r.value, tb) for r, tb in opponent_keys]
    
            all_keys = [hero_key] + opponent_keys
            best_key = max(all_keys)
            winners = [k for k in all_keys if k == best_key]
    
            if hero_key == best_key:
                if len(winners) == 1:
                    wins += 1
                else:
                    ties += 1
            else:
                losses += 1
    
        total = wins + ties + losses
        return {
            'loss_probability': losses / total,
            'tie_probability': ties / total, 
            'win_probability': wins / total,
            'samples': total
        }
    


            
    
    def predict(self, hole_cards: List[str], board_cards: List[str], num_players: int) -> Dict[str, float]:
        """
        Predict win/tie/loss probabilities for player 0 using ML model.
    
        Parameters
        ----------
        hole_cards : List[str]
            Two hole cards for player 0 (e.g., ['As', 'Kh']).
        board_cards : List[str]
            Up to 5 board cards (e.g., ['2d', '5h', '9s']).
        num_players : int
            Number of players in the game.
    
        Returns
        -------
        Dict[str, float]
            Dictionary with loss/tie/win probabilities and predicted class.
        """
        logger.info(f'HOLE CARD INPUTS: {hole_cards}')
        logger.info(f'BOARD CARD INPUTS: {board_cards}')
        logger.info(f'NUM_PLAYERS INPUT: {num_players}')
        if len(hole_cards) != 2:
            raise ValueError("Exactly 2 hole cards required for player 0.")
        if len(board_cards) > 5:
            raise ValueError("Board cannot have more than 5 cards.")
            
        hole_tuples = [parse_card(c) for c in hole_cards]
        board_tuples = [parse_card(c) for c in board_cards]
        features = encode_game_state(hole_tuples, board_tuples, num_players)
        features = features.reshape(1, -1)
    
        prediction = self.model.predict(features, verbose=0)[0]
        return {
            'loss_probability': float(prediction[0]),
            'tie_probability': float(prediction[1]),
            'win_probability': float(prediction[2]),
            'predicted_class': ['Loss', 'Tie', 'Win'][np.argmax(prediction)]
        }




    def _create_cards(self, card_strs: List[str]) -> List[Card]:
        """
        Convert string card representations to Card objects.

        Parameters
        ----------
        card_strs : List[str]
            List of strings like ['As', 'Tc'].

        Returns
        -------
        List[Card]
            List of Card objects.
        """
        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
            'K': 13, 'A': 14
        }
        suit_map = {
            'c': Suit.CLUBS,
            'd': Suit.DIAMONDS,
            'h': Suit.HEARTS,
            's': Suit.SPADES
        }

        cards = []
        for s in card_strs:
            if len(s) != 2:
                raise ValueError(f"Invalid card string: '{s}'")
            rank = rank_map.get(s[0].upper())
            suit = suit_map.get(s[1].lower())
            if rank is None or suit is None:
                raise ValueError(f"Invalid card string: '{s}'")
            cards.append(Card(rank, suit))
        return cards
