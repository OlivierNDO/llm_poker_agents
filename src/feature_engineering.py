"""
feature_engineering.py

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

def encode_game_state_from_strings(hole_cards: list, board_cards: list, num_players : int):
	hole_tuples = [parse_card(c) for c in hole_cards]
	board_tuples = [parse_card(c) for c in board_cards]
	features = encode_game_state(hole_tuples, board_tuples, num_players)
	#features = features.reshape(1, -1)
	return features



x2 = encode_game_state_from_strings(
    hole_cards = ['Ad', 'Kd'],
    board_cards = ['2d', '7d', '9d'],
    num_players = 6
)

x2.shape

"""
from typing import List

from numba import njit
import numpy as np

@njit
def extract_flush_features(hole_suits: np.ndarray, board_suits: np.ndarray) -> np.ndarray:
    """
    FIXED: Now properly includes hole cards in suit counting
    """
    suited_hole = 1.0 if hole_suits[0] == hole_suits[1] else 0.0

    # FIXED: Include both hole and board cards in suit counting
    suit_counts = np.zeros(4, dtype=np.int32)
    
    # Add hole cards to suit counts
    for s in hole_suits:
        if s >= 0:  # Valid suit
            suit_counts[s] += 1
            
    # Add board cards to suit counts
    for s in board_suits:
        if s >= 0:  # Skip zero padding
            suit_counts[s] += 1
            
    board_2flush = 0.0
    board_3flush = 0.0
    board_4flush = 0.0
    board_5flush = 0.0
    flush_possible_with_hole = 0.0
    backdoor_flush_draw = 0.0
    flush_blocker = 0.0

    for suit in range(4):
        total_count = suit_counts[suit]
        board_count = 0
        hole_count = 0
        
        # Count board cards of this suit
        for s in board_suits:
            if s == suit:
                board_count += 1
                
        # Count hole cards of this suit  
        for s in hole_suits:
            if s == suit:
                hole_count += 1
        
        # Board flush indicators (board cards only)
        if board_count == 2:
            board_2flush = 1.0
            if hole_count >= 1:  # Need 1+ hole cards of same suit for backdoor
                backdoor_flush_draw = 1.0
        elif board_count == 3:
            board_3flush = 1.0
        elif board_count == 4:
            board_4flush = 1.0
        elif board_count >= 5:
            board_5flush = 1.0

        # Flush possible with hole cards (total count)
        if total_count >= 5:
            flush_possible_with_hole = 1.0
            if hole_count > 0:
                flush_blocker = 1.0

    return np.array([
        suited_hole,
        board_2flush,
        board_3flush,
        board_4flush,
        board_5flush,
        flush_possible_with_hole,
        backdoor_flush_draw,
        flush_blocker
    ], dtype=np.float32)



@njit
def extract_rank_features(hole_ranks: np.ndarray, board_ranks: np.ndarray) -> np.ndarray:
    all_ranks = np.zeros(15, dtype=np.int32)  # 2-14 inclusive
    for r in hole_ranks:
        all_ranks[r] += 1
    for r in board_ranks:
        if r > 1:  # Skip zero padding
            all_ranks[r] += 1

    pair_rank = 0
    second_pair_rank = 0
    trips_rank = 0
    quads_rank = 0
    pair_from_pocket = 0
    trips_from_pocket = 0
    quads_from_pocket = 0

    # Find all pairs first
    pairs_found = []
    for rank in range(2, 15):
        count = all_ranks[rank]
        if count == 2:
            pairs_found.append(rank)
        if count == 3 and rank > trips_rank:
            trips_rank = rank
        if count == 4 and rank > quads_rank:
            quads_rank = rank

    # Set pair ranks
    if len(pairs_found) >= 1:
        pairs_found.sort(reverse=True)  # Highest first
        pair_rank = pairs_found[0]
        if len(pairs_found) >= 2:
            second_pair_rank = pairs_found[1]

    # Kicker calculation
    pair_kicker = 0
    trips_kicker = 0
    
    if pair_rank > 0:
        # Find highest non-paired card for pair kicker
        for rank in range(14, 1, -1):
            if all_ranks[rank] == 1:  # Single card, not paired
                pair_kicker = rank
                break
    
    if trips_rank > 0:
        # Find highest non-trips card for trips kicker
        for rank in range(14, 1, -1):
            if all_ranks[rank] == 1 or (all_ranks[rank] == 2 and rank != trips_rank):
                trips_kicker = rank
                break

    # Pocket involvement
    if hole_ranks[0] == hole_ranks[1]:
        if all_ranks[hole_ranks[0]] >= 2:
            pair_from_pocket = 1
        if all_ranks[hole_ranks[0]] >= 3:
            trips_from_pocket = 1
        if all_ranks[hole_ranks[0]] == 4:
            quads_from_pocket = 1

    return np.array([
        pair_rank / 14.0, float(pair_from_pocket),
        second_pair_rank / 14.0,  # NEW: Second pair rank
        trips_rank / 14.0, float(trips_from_pocket),
        quads_rank / 14.0, float(quads_from_pocket),
        pair_kicker / 14.0,  # NEW: Pair kicker
        trips_kicker / 14.0  # NEW: Trips kicker
    ], dtype=np.float32)




@njit
def extract_contextual_features(hole_ranks: np.ndarray, board_ranks: np.ndarray) -> np.ndarray:
    """
    FIXED: Proper gap calculation handling A-2 connectivity
    """
    hole_pair = 1.0 if hole_ranks[0] == hole_ranks[1] else 0.0

    max_board_rank = 0
    for r in board_ranks:
        if r > max_board_rank:
            max_board_rank = r

    overcards = 1.0 if (hole_ranks[0] > max_board_rank and hole_ranks[1] > max_board_rank) else 0.0
    kicker_rank = max(hole_ranks[0], hole_ranks[1]) / 14.0
    top_board = max_board_rank / 14.0

    above_board = 0
    for r in hole_ranks:
        if r > max_board_rank:
            above_board += 1

    # Hand improvement logic (unchanged - this was correct)
    rank_counts_full = np.zeros(15, dtype=np.int32)
    rank_counts_board = np.zeros(15, dtype=np.int32)

    for r in board_ranks:
        rank_counts_board[r] += 1
        rank_counts_full[r] += 1
    for r in hole_ranks:
        rank_counts_full[r] += 1

    def get_best_rank(counts):
        for rank in range(14, 1, -1):
            if counts[rank] == 4:
                return 7
            elif counts[rank] == 3:
                for r2 in range(14, 1, -1):
                    if r2 != rank and counts[r2] >= 2:
                        return 6
                return 3
            elif counts[rank] == 2:
                for r2 in range(rank - 1, 1, -1):
                    if counts[r2] == 2:
                        return 2
                return 1
        return 0

    board_strength = get_best_rank(rank_counts_board)
    full_strength = get_best_rank(rank_counts_full)
    hole_improves_board = 1.0 if full_strength > board_strength else 0.0
    
    # FIXED: Proper gap calculation with A-2 connectivity
    r1, r2 = hole_ranks[0], hole_ranks[1]
    
    # Calculate gap considering A-2 connectivity
    if (r1 == 14 and r2 == 2) or (r1 == 2 and r2 == 14):
        gap = 0  # A-2 is connected
    elif r1 == 14 and r2 == 3:
        gap = 1  # A-3 has one gap (missing 2)
    elif r1 == 3 and r2 == 14:
        gap = 1  # Same as above
    else:
        gap = abs(r1 - r2) - 1
    
    gap = max(0, gap)
    gap_normalized = min(gap, 12) / 12.0

    return np.array([
        hole_pair,
        overcards,
        kicker_rank,
        top_board,
        float(above_board),
        hole_improves_board,
        gap_normalized
    ], dtype=np.float32)



@njit
def extract_board_features(hole_ranks: np.ndarray, board_ranks: np.ndarray) -> np.ndarray:
    """
    FIXED: Proper full house detection ensuring trips and pair are different ranks
    """
    board_rank_counts = np.zeros(15, dtype=np.int32)
    for r in board_ranks:
        if r > 1:
            board_rank_counts[r] += 1
    
    hole_rank_counts = np.zeros(15, dtype=np.int32)
    for r in hole_ranks:
        hole_rank_counts[r] += 1
    
    board_paired = 0
    board_trips = 0
    board_quads = 0
    board_full_house = 0
    hole_beats_board_pair = 0
    hole_makes_full_house = 0
    
    # FIXED: Track specific ranks for proper full house detection
    trips_ranks = []
    pair_ranks = []
    
    for rank in range(2, 15):
        count = board_rank_counts[rank]
        if count == 2:
            board_paired = 1
            pair_ranks.append(rank)
        elif count == 3:
            board_trips = 1
            trips_ranks.append(rank)
        elif count == 4:
            board_quads = 1
            trips_ranks.append(rank)  # Quads contain trips
            pair_ranks.append(rank)   # Quads also contain pairs
    
    # FIXED: Board has full house only if trips and pair are different ranks
    if len(trips_ranks) > 0 and len(pair_ranks) > 0:
        # Check if any trips rank is different from any pair rank
        has_different_ranks = False
        for trips_rank in trips_ranks:
            for pair_rank in pair_ranks:
                if trips_rank != pair_rank:
                    has_different_ranks = True
                    break
            if has_different_ranks:
                break
        
        if has_different_ranks:
            board_full_house = 1
    
    # Highest board pair
    top_board_pair = 0
    for r in range(14, 1, -1):
        if board_rank_counts[r] >= 2:
            top_board_pair = r
            break
    
    for r in hole_ranks:
        if r > top_board_pair:
            hole_beats_board_pair = 1
            break
    
    # Full house with hole cards
    combined = board_rank_counts + hole_rank_counts
    trips_count = 0
    pair_count = 0
    for rank in range(2, 15):
        count = combined[rank]
        if count >= 3:
            trips_count += 1
        if count >= 2:
            pair_count += 1
    
    # Need trips AND separate pair (pair_count > trips_count means extra pairs)
    if trips_count >= 1 and pair_count > trips_count:
        hole_makes_full_house = 1
    
    return np.array([
        float(board_paired),
        float(board_trips),
        float(board_quads),
        float(board_full_house),
        float(hole_beats_board_pair),
        float(hole_makes_full_house)
    ], dtype=np.float32)


@njit
def extract_interaction_features(hole_suits: np.ndarray,
                                 hole_ranks: np.ndarray,
                                 board_suits: np.ndarray,
                                 board_ranks: np.ndarray,
                                 flush_possible: float,
                                 backdoor_flush: float,
                                 straight_possible: float,
                                 open_ended: float,
                                 gutshot: float) -> np.ndarray:
    """
    NEW: Extract interaction features between different poker concepts
    """
    # 1. Flush + Straight Draw Interaction (CRITICAL)
    has_straight_draw = 1.0 if (open_ended or gutshot) else 0.0
    flush_and_straight_draw = 1.0 if (flush_possible and has_straight_draw) else 0.0
    backdoor_and_straight_draw = 1.0 if (backdoor_flush and has_straight_draw) else 0.0

    # 2. Nut Draw Indicators (fixed: include hole cards in suit counts)
    # recompute suit counts including both board and hole
    suit_counts = np.zeros(4, dtype=np.int32)
    for s in board_suits:
        if s >= 0:
            suit_counts[s] += 1
    for s in hole_suits:
        if s >= 0:
            suit_counts[s] += 1

    nut_flush_draw = 0.0
    nut_backdoor_flush = 0.0
    for suit in range(4):
        count = suit_counts[suit]
        has_ace = ((hole_ranks[0] == 14 and hole_suits[0] == suit) or
                   (hole_ranks[1] == 14 and hole_suits[1] == suit))
        if has_ace:
            if count >= 4:           # four to a nut flush
                nut_flush_draw = 1.0
            elif count == 3:         # backdoor nut flush
                nut_backdoor_flush = 1.0
            break

    # 3. Nut straight draw = open ended with high cards
    nut_straight_draw = 0.0
    if open_ended:
        if max(hole_ranks[0], hole_ranks[1]) >= 10:
            nut_straight_draw = 1.0

    # 4. Two Pair vs Set Vulnerability
    board_has_pair = 0.0
    for r in board_ranks:
        if r > 1:
            if (board_ranks == r).sum() >= 2:
                board_has_pair = 1.0
                break

    all_ranks = np.zeros(15, dtype=np.int32)
    for r in hole_ranks:
        all_ranks[r] += 1
    for r in board_ranks:
        if r > 1:
            all_ranks[r] += 1

    pair_count = (all_ranks == 2).sum()
    two_pair_vs_set_vulnerability = 1.0 if (pair_count >= 2 and board_has_pair) else 0.0

    # 5. Counterfeiting Risk
    counterfeiting_risk = 0.0
    if pair_count >= 1:
        lowest_pair = 15
        for r in range(2, 15):
            if all_ranks[r] == 2 and r < lowest_pair:
                lowest_pair = r
        for r in board_ranks:
            if 1 < r < lowest_pair and (board_ranks == r).sum() >= 2:
                counterfeiting_risk = 1.0
                break

    return np.array([
        flush_and_straight_draw,
        backdoor_and_straight_draw,
        nut_flush_draw,
        nut_backdoor_flush,
        nut_straight_draw,
        two_pair_vs_set_vulnerability,
        counterfeiting_risk
    ], dtype=np.float32)



@njit
def extract_simple_board_texture(board_ranks: np.ndarray) -> float:
    """Super simple board texture measure"""
    active_ranks = []
    for r in board_ranks:
        if r > 1:
            active_ranks.append(r)
    
    if len(active_ranks) < 3:
        return 0.0
    
    active_ranks.sort()
    span = active_ranks[-1] - active_ranks[0]
    
    # Tight span = coordinated board, wide span = dry board
    return max(0.0, 1.0 - span / 12.0)  # 0 = very dry, 1 = very coordinated





HAND_NAMES = (
    'high_card',
    'one_pair',
    'two_pair',
    'three_of_a_kind',
    'straight',
    'flush',
    'full_house',
    'four_of_a_kind',
    'straight_flush',
)

def best_hand_name(category_code: int) -> str:
    return HAND_NAMES[int(category_code)]

@njit
def evaluate_best_hand(hole_ranks: np.ndarray,
                       board_ranks: np.ndarray,
                       hole_suits: np.ndarray,
                       board_suits: np.ndarray) -> tuple:
    """
    Returns (category_code, tiebreakers[5]) where category_code is:
      0=high_card, 1=one_pair, 2=two_pair, 3=three_of_a_kind,
      4=straight, 5=flush, 6=full_house, 7=four_of_a_kind, 8=straight_flush.
    Tiebreakers are category-specific rank keys, descending.
    """
    all_ranks = np.zeros(15, np.int32)
    for r in hole_ranks:
        if r > 1:
            all_ranks[r] += 1
    for r in board_ranks:
        if r > 1:
            all_ranks[r] += 1

    suit_counts = np.zeros(4, np.int32)
    for i in range(hole_suits.shape[0]):
        if hole_ranks[i] > 1 and hole_suits[i] >= 0:
            suit_counts[hole_suits[i]] += 1
    for i in range(board_suits.shape[0]):
        if board_ranks[i] > 1 and board_suits[i] >= 0:
            suit_counts[board_suits[i]] += 1

    has_sf = 0
    sf_high = 0
    for suit in range(4):
        if suit_counts[suit] >= 5:
            suited = np.zeros(15, np.uint8)
            for i in range(hole_suits.shape[0]):
                if hole_suits[i] == suit and hole_ranks[i] > 1:
                    suited[hole_ranks[i]] = 1
                    if hole_ranks[i] == 14:
                        suited[1] = 1
            for i in range(board_suits.shape[0]):
                if board_suits[i] == suit and board_ranks[i] > 1:
                    suited[board_ranks[i]] = 1
                    if board_ranks[i] == 14:
                        suited[1] = 1
            for low in range(1, 11):
                if np.sum(suited[low:low+5]) == 5:
                    has_sf = 1
                    sf_high = 5 if low == 1 else low + 4
                    break
            if has_sf == 1:
                break

    if has_sf == 1:
        return 8, np.array([sf_high, 0, 0, 0, 0], np.int32)

    quads = 0
    trips = 0
    pairs = np.zeros(5, np.int32)
    pc = 0
    for rank in range(14, 1, -1):
        c = all_ranks[rank]
        if c == 4:
            quads = rank
        elif c == 3 and trips == 0:
            trips = rank
        elif c == 2:
            if pc < 5:
                pairs[pc] = rank
                pc += 1

    if quads > 0:
        kicker = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                kicker = r
                break
        return 7, np.array([quads, kicker, 0, 0, 0], np.int32)

    has_flush = 0
    flush_suit = -1
    for suit in range(4):
        if suit_counts[suit] >= 5:
            has_flush = 1
            flush_suit = suit
            break

    has_straight = 0
    straight_high = 0
    flags = np.zeros(15, np.uint8)
    for r in range(14, 1, -1):
        if all_ranks[r] > 0:
            flags[r] = 1
            if r == 14:
                flags[1] = 1
    for low in range(1, 11):
        if np.sum(flags[low:low+5]) == 5:
            has_straight = 1
            straight_high = 5 if low == 1 else low + 4
            break

    if trips > 0 and pc > 0:
        return 6, np.array([trips, pairs[0], 0, 0, 0], np.int32)

    if has_flush == 1:
        flush_cards = np.zeros(7, np.int32)
        fc = 0
        for i in range(hole_suits.shape[0]):
            if hole_suits[i] == flush_suit and hole_ranks[i] > 1:
                flush_cards[fc] = hole_ranks[i]
                fc += 1
        for i in range(board_suits.shape[0]):
            if board_suits[i] == flush_suit and board_ranks[i] > 1:
                flush_cards[fc] = board_ranks[i]
                fc += 1
        for i in range(fc - 1):
            for j in range(i + 1, fc):
                if flush_cards[j] > flush_cards[i]:
                    tmp = flush_cards[i]
                    flush_cards[i] = flush_cards[j]
                    flush_cards[j] = tmp
        while fc < 5:
            flush_cards[fc] = 0
            fc += 1
        return 5, np.array([flush_cards[0], flush_cards[1], flush_cards[2], flush_cards[3], flush_cards[4]], np.int32)

    if has_straight == 1:
        return 4, np.array([straight_high, 0, 0, 0, 0], np.int32)

    if trips > 0:
        kick1 = 0
        kick2 = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                if kick1 == 0:
                    kick1 = r
                elif kick2 == 0:
                    kick2 = r
                    break
        return 3, np.array([trips, kick1, kick2, 0, 0], np.int32)

    if pc >= 2:
        pair_hi = pairs[0]
        pair_lo = pairs[1]
        kicker = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                kicker = r
                break
        return 2, np.array([pair_hi, pair_lo, kicker, 0, 0], np.int32)

    if pc == 1:
        pair_rank = pairs[0]
        k1 = 0
        k2 = 0
        k3 = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                if k1 == 0:
                    k1 = r
                elif k2 == 0:
                    k2 = r
                elif k3 == 0:
                    k3 = r
                    break
        return 1, np.array([pair_rank, k1, k2, k3, 0], np.int32)

    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0
    k5 = 0
    kc = 0
    for r in range(14, 1, -1):
        if all_ranks[r] >= 1:
            if kc == 0:
                k1 = r
            elif kc == 1:
                k2 = r
            elif kc == 2:
                k3 = r
            elif kc == 3:
                k4 = r
            elif kc == 4:
                k5 = r
                break
            kc += 1
    return 0, np.array([k1, k2, k3, k4, k5], np.int32)



def best_hand_string(hole_ranks: np.ndarray,
                     board_ranks: np.ndarray,
                     hole_suits: np.ndarray,
                     board_suits: np.ndarray) -> str:
    """
    Wrapper: evaluate the best hand and return a descriptive string.
    """
    code, tiebreakers = evaluate_best_hand(hole_ranks, board_ranks, hole_suits, board_suits)

    rank_names = {
        14: "Ace", 13: "King", 12: "Queen", 11: "Jack",
        10: "Ten", 9: "Nine", 8: "Eight", 7: "Seven",
        6: "Six", 5: "Five", 4: "Four", 3: "Three", 2: "Two"
    }

    if code == 0:   # high card
        return f"High card {rank_names[tiebreakers[0]]}"
    elif code == 1: # one pair
        return f"Pair of {rank_names[tiebreakers[0]]}s"
    elif code == 2: # two pair
        return f"Two pair: {rank_names[tiebreakers[0]]}s and {rank_names[tiebreakers[1]]}s"
    elif code == 3: # trips
        return f"Three of a kind: {rank_names[tiebreakers[0]]}s"
    elif code == 4: # straight
        return f"Straight to the {rank_names[tiebreakers[0]]}"
    elif code == 5: # flush
        return f"Flush, high card {rank_names[tiebreakers[0]]}"
    elif code == 6: # full house
        return f"Full house: {rank_names[tiebreakers[0]]}s full of {rank_names[tiebreakers[1]]}s"
    elif code == 7: # quads
        return f"Four of a kind: {rank_names[tiebreakers[0]]}s"
    elif code == 8: # straight flush
        return f"Straight flush to the {rank_names[tiebreakers[0]]}"
    else:
        return "Unknown hand"




@njit
def extract_hand_strength_scalar(hole_ranks: np.ndarray, board_ranks: np.ndarray, 
                                hole_suits: np.ndarray, board_suits: np.ndarray) -> float:
    """
    FIXED: Consistent suit counting and proper invalid card handling
    """
    all_ranks = np.zeros(15, dtype=np.int32)
    for r in hole_ranks:
        if r > 1:  # FIXED: Consistent invalid card check
            all_ranks[r] += 1
    for r in board_ranks:
        if r > 1:
            all_ranks[r] += 1
    
    # FIXED: Consistent suit counting
    suit_counts = np.zeros(4, dtype=np.int32)
    
    # Add hole cards to suit counts
    for i in range(len(hole_suits)):
        if hole_suits[i] >= 0 and hole_ranks[i] > 1:  # Valid card
            suit_counts[hole_suits[i]] += 1
    
    # Add board cards to suit counts
    for i in range(len(board_suits)):
        if i < len(board_suits) and board_suits[i] >= 0 and board_ranks[i] > 1:
            suit_counts[board_suits[i]] += 1
    
    # Straight flush detection (unchanged - this was correct)
    has_straight_flush = 0
    straight_flush_high = 0
    
    for suit in range(4):
        if suit_counts[suit] >= 5:
            suited_rank_flags = np.zeros(15, dtype=np.uint8)
            
            # Add hole cards of this suit
            for i in range(len(hole_suits)):
                if hole_suits[i] == suit and hole_ranks[i] > 1:
                    suited_rank_flags[hole_ranks[i]] = 1
                    if hole_ranks[i] == 14:
                        suited_rank_flags[1] = 1
            
            # Add board cards of this suit
            for i in range(len(board_suits)):
                if i < len(board_suits) and board_suits[i] == suit and board_ranks[i] > 1:
                    suited_rank_flags[board_ranks[i]] = 1
                    if board_ranks[i] == 14:
                        suited_rank_flags[1] = 1
            
            # Check for straight within this suit
            for low in range(1, 11):
                if np.sum(suited_rank_flags[low:low+5]) == 5:
                    has_straight_flush = 1
                    straight_flush_high = low + 4 if low > 1 else 5
                    break
            
            if has_straight_flush:
                break
    
    # Regular flush detection
    has_flush = 0
    flush_high = 0
    for suit in range(4):
        if suit_counts[suit] >= 5:
            has_flush = 1
            # Find highest card in this flush suit
            for i in range(len(hole_suits)):
                if hole_suits[i] == suit and hole_ranks[i] > 1:
                    flush_high = max(flush_high, hole_ranks[i])
            for i in range(len(board_suits)):
                if i < len(board_suits) and board_suits[i] == suit and board_ranks[i] > 1:
                    flush_high = max(flush_high, board_ranks[i])
            break
    
    # Regular straight detection
    rank_flags = np.zeros(15, dtype=np.uint8)
    for r in hole_ranks:
        if r > 1:
            rank_flags[r] = 1
            if r == 14:
                rank_flags[1] = 1
    for r in board_ranks:
        if r > 1:
            rank_flags[r] = 1
            if r == 14:
                rank_flags[1] = 1
    
    has_straight = 0
    straight_high = 0
    for low in range(1, 11):
        if np.sum(rank_flags[low:low+5]) == 5:
            has_straight = 1
            straight_high = low + 4 if low > 1 else 5
            break
    
    # Hand ranking logic (unchanged - this was correct)
    quads = 0
    trips = 0
    pairs_found = []
    
    for rank in range(2, 15):
        count = all_ranks[rank]
        if count == 4:
            quads = rank
        elif count == 3:
            trips = rank
        elif count == 2:
            pairs_found.append(rank)
    
    pairs_found.sort(reverse=True)
    pair1 = pairs_found[0] if len(pairs_found) >= 1 else 0
    pair2 = pairs_found[1] if len(pairs_found) >= 2 else 0
    
    # Calculate strength
    strength = 0.0
    
    if has_straight_flush:
        strength = 800.0 + straight_flush_high * 1.0
    elif quads > 0:
        kicker = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                kicker = r
                break
        strength = 700.0 + quads * 6.0 + kicker * 0.1
    elif trips > 0 and pair1 > 0:
        strength = 600.0 + trips * 5.0 + pair1 * 0.1
    elif has_flush:
        flush_cards = []
        flush_suit = 0
        
        for suit in range(4):
            if suit_counts[suit] >= 5:
                flush_suit = suit
                break
        
        # Collect flush cards
        for i in range(len(hole_suits)):
            if hole_suits[i] == flush_suit and hole_ranks[i] > 1:
                flush_cards.append(hole_ranks[i])
        for i in range(len(board_suits)):
            if i < len(board_suits) and board_suits[i] == flush_suit and board_ranks[i] > 1:
                flush_cards.append(board_ranks[i])
        
        flush_cards.sort(reverse=True)
        flush_cards = flush_cards[:5]
        second_flush = flush_cards[1] if len(flush_cards) > 1 else 0
        third_flush = flush_cards[2] if len(flush_cards) > 2 else 0
        
        strength = 500.0 + flush_cards[0] * 6.0 + second_flush * 0.6 + third_flush * 0.06
    elif has_straight:
        strength = 400.0 + straight_high * 6.0
    elif trips > 0:
        kicker = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                kicker = r
                break
        
        set_bonus = 0.0
        if hole_ranks[0] == hole_ranks[1] and hole_ranks[0] == trips:
            set_bonus = 0.5
        
        strength = 300.0 + trips * 5.0 + kicker * 0.1 + set_bonus
    elif len(pairs_found) >= 2:
        kicker = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                kicker = r
                break
        strength = 200.0 + pair1 * 6.0 + pair2 * 0.5 + kicker * 0.01
    elif pair1 > 0:
        kicker = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                kicker = r
                break
        strength = 100.0 + pair1 * 6.0 + kicker * 0.1
    else:
        high = 0
        second = 0
        for r in hole_ranks:
            if r > 1:
                if r > high:
                    second = high
                    high = r
                elif r > second:
                    second = r
        strength = high * 6.0 + second * 0.1
    
    return min(1.0, strength / 814.0)



@njit
def extract_dominated_draws(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray,
    flush_possible: float,
    straight_possible: float
) -> np.ndarray:
    # Flush: dominated if no ace in hole of flush suit
    dominated_flush = 0.0
    if flush_possible:
        flush_suit = hole_suits[0] if hole_suits[0] == hole_suits[1] else -1
        if flush_suit >= 0:
            has_ace = (hole_ranks[0] == 14 and hole_suits[0] == flush_suit) or \
                      (hole_ranks[1] == 14 and hole_suits[1] == flush_suit)
            if not has_ace:
                dominated_flush = 1.0

    # Straight: dominated if high card is low
    dominated_straight = 0.0
    if straight_possible and max(hole_ranks[0], hole_ranks[1]) < 12:
        dominated_straight = 1.0

    return np.array([dominated_flush, dominated_straight], dtype=np.float32)


@njit
def extract_board_volume_features(board_ranks: np.ndarray, board_suits: np.ndarray) -> np.ndarray:
    connected = 0
    broadway = 0
    suit_counts = np.zeros(4, dtype=np.int32)

    for i in range(5):
        r = board_ranks[i]
        if r > 1:
            if r >= 10:
                broadway += 1
            for j in range(5):
                if i != j and abs(r - board_ranks[j]) == 1:
                    connected += 1
            suit = board_suits[i]
            if suit >= 0:
                suit_counts[suit] += 1

    connected_pairs = connected // 2  # Each pair counted twice
    max_suit = 0
    for s in range(4):
        if suit_counts[s] > max_suit:
            max_suit = suit_counts[s]

    return np.array([float(connected_pairs), float(broadway), float(max_suit)], dtype=np.float32)


@njit
def extract_flush_blocker_count(hole_suits: np.ndarray, board_suits: np.ndarray) -> np.ndarray:
    suit_counts = np.zeros(4, dtype=np.int32)
    for s in board_suits:
        if s >= 0:
            suit_counts[s] += 1

    dominant_suit = -1
    max_count = 0
    for s in range(4):
        if suit_counts[s] > max_count:
            dominant_suit = s
            max_count = suit_counts[s]

    blocker_count = 0
    for s in hole_suits:
        if s == dominant_suit:
            blocker_count += 1

    return np.array([float(blocker_count)], dtype=np.float32)



@njit
def extract_advanced_poker_features(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray,
    board_ranks: np.ndarray,
    board_suits: np.ndarray
) -> np.ndarray:
    """
    Extract advanced features including:
    - Normalized kicker strength
    - Paired board strength
    - Full house draw potential
    - Flush strength estimate
    - Trips board kicker dominance

    Returns
    -------
    np.ndarray
        Array of 5 float32 features
    """
    # Normalize kicker
    kicker_strength = 0.0
    if hole_ranks[0] != hole_ranks[1]:
        kicker_strength = min(hole_ranks[0], hole_ranks[1]) / 14.0

    # Paired board strength
    rank_counts = np.zeros(15, dtype=np.int32)
    for r in board_ranks:
        if r > 0:
            rank_counts[r] += 1
    paired_board_rank = 0
    for r in range(2, 15):
        if rank_counts[r] >= 2:
            paired_board_rank = r
            break
    paired_strength = paired_board_rank / 14.0 if paired_board_rank else 0.0

    # Full house draw from trips + pocket pair
    board_trips_rank = -1
    for r in range(2, 15):
        if rank_counts[r] == 3:
            board_trips_rank = r
            break
    full_house_draw = 0.0
    if board_trips_rank >= 0 and hole_ranks[0] == hole_ranks[1]:
        full_house_draw = 1.0

    # Flush strength estimate
    suited = hole_suits[0] == hole_suits[1]
    flush_strength = 0.0
    if suited:
        flush_strength = max(hole_ranks[0], hole_ranks[1]) / 14.0

    # Trips board kicker beat
    beats_trip_kicker = 0.0
    if board_trips_rank >= 0:
        high_hole = max(hole_ranks[0], hole_ranks[1])
        if high_hole > board_trips_rank:
            beats_trip_kicker = 1.0

    return np.array([
        kicker_strength,
        paired_strength,
        full_house_draw,
        flush_strength,
        beats_trip_kicker
    ], dtype=np.float32)


@njit
def extract_relative_board_contribution(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray,
    board_ranks: np.ndarray,
    board_suits: np.ndarray
) -> np.ndarray:
    """
    FIXED: Use consistent invalid card values
    """
    strength_full = extract_hand_strength_scalar(hole_ranks, board_ranks, hole_suits, board_suits)

    # FIXED: Use 0 for invalid cards (consistent with other checks)
    blank_hole_ranks = np.array([0, 0], dtype=np.int32)
    blank_hole_suits = np.array([-1, -1], dtype=np.int32)
    
    strength_board = extract_hand_strength_scalar(blank_hole_ranks, board_ranks, blank_hole_suits, board_suits)

    denom = strength_full if strength_full > 1e-6 else 1e-6
    relative = strength_board / denom
    return np.array([min(1.0, relative)], dtype=np.float32)


@njit
def extract_straight_features(hole_ranks: np.ndarray, board_ranks: np.ndarray) -> np.ndarray:
    """
    FIXED: Better bounds checking, consistent invalid card handling, and proper straight detection
    """
    # Create rank_flags for all cards (hole + board) with consistent invalid card handling
    rank_flags = np.zeros(15, dtype=np.uint8)
    
    # Add hole cards with proper invalid card check
    for r in hole_ranks:
        if r > 1:  # FIXED: Consistent with other functions (skip 0 and 1 as invalid)
            rank_flags[r] = 1
            if r == 14:  # Ace can be low in straights
                rank_flags[1] = 1
    
    # Add board cards with bounds checking
    for i in range(min(5, len(board_ranks))):  # FIXED: Prevent index out of bounds
        r = board_ranks[i]
        if r > 1:  # FIXED: Consistent invalid card check
            rank_flags[r] = 1
            if r == 14:  # Ace can be low in straights
                rank_flags[1] = 1

    # Create board_flags for board-only features with same fixes
    board_flags = np.zeros(15, dtype=np.uint8)
    for i in range(min(5, len(board_ranks))):  # FIXED: Bounds checking
        r = board_ranks[i]
        if r > 1:  # FIXED: Consistent invalid card check
            board_flags[r] = 1
            if r == 14:  # Ace can be low
                board_flags[1] = 1

    # Board straight detection (board cards only)
    board_3straight = 0
    board_4straight = 0
    board_5straight = 0
    
    for low in range(1, 11):  # Check all possible straight starting positions (A-low to 10-high)
        span = board_flags[low:low+5]
        total = np.sum(span)
        if total >= 3:
            board_3straight = 1
        if total >= 4:
            board_4straight = 1
        if total == 5:
            board_5straight = 1
            break  # Found board straight, no need to check higher

    # Straight possible with hole cards (all cards)
    straight_possible_with_hole = 0
    for low in range(1, 11):  # A-2-3-4-5 through 10-J-Q-K-A
        if np.sum(rank_flags[low:low+5]) == 5:
            straight_possible_with_hole = 1
            break

    # Draw detection (4 cards to straight)
    open_ended = 0
    gutshot = 0
    
    for low in range(1, 11):
        window = rank_flags[low:low+5]
        total = np.sum(window)
        
        if total == 4:  # Missing exactly one card for straight
            # Find which position is missing
            missing_position = -1
            for i in range(5):
                if window[i] == 0:
                    missing_position = i
                    break
            
            # Open-ended: missing first or last card (positions 0 or 4)
            # Gutshot: missing any middle card (positions 1, 2, or 3)
            if missing_position == 0 or missing_position == 4:
                open_ended = 1
            else:
                gutshot = 1

    return np.array([
        float(board_3straight),          # Board has 3+ cards to straight
        float(board_4straight),          # Board has 4+ cards to straight  
        float(board_5straight),          # Board has complete straight
        float(straight_possible_with_hole), # Hole + board make straight
        float(open_ended),               # Open-ended straight draw
        float(gutshot)                   # Gutshot straight draw
    ], dtype=np.float32)



from numba import njit
import numpy as np

@njit
def calculate_pair_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    card_rank: int
) -> float:
    """
    Numba-optimized version for low latency.
    """
    # Fast counting with early exit
    my_count = 0
    board_count = 0
    
    # Count hole cards
    for i in range(len(hole_ranks)):
        if hole_ranks[i] == card_rank:
            my_count += 1
            if my_count == 2:
                return 1.0  # Early exit: already have pair
    
    if my_count == 0:
        return 0.0  # Early exit: can't make pair
    
    # Count board cards
    for i in range(len(board_ranks)):
        if board_ranks[i] == card_rank and board_ranks[i] > 1:  # Skip padding
            board_count += 1
    
    if my_count + board_count >= 2:
        return 1.0  # Already have pair
    
    # Fast remaining calculation
    cards_remaining_of_rank = 4 - my_count - board_count
    if cards_remaining_of_rank == 0:
        return 0.0
    
    # Count actual board cards (skip padding)
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
    
    board_cards_remaining = 5 - actual_board_cards
    if board_cards_remaining == 0:
        return 0.0
    
    remaining_deck_size = 52 - len(hole_ranks) - actual_board_cards
    non_rank_remaining = remaining_deck_size - cards_remaining_of_rank
    
    # Optimized probability calculation
    if board_cards_remaining == 1:
        return float(cards_remaining_of_rank) / float(remaining_deck_size)
    elif board_cards_remaining == 2:
        prob_no_rank = (float(non_rank_remaining) * float(non_rank_remaining - 1)) / (float(remaining_deck_size) * float(remaining_deck_size - 1))
        return 1.0 - prob_no_rank
    else:
        prob_no_rank = 1.0
        for i in range(board_cards_remaining):
            prob_no_rank *= float(non_rank_remaining - i) / float(remaining_deck_size - i)
        return 1.0 - prob_no_rank



@njit
def calculate_opponent_pair_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    card_rank: int,
    num_opponents: int
) -> float:
    """
    Full calculation using basic probability and combinatorics (no loops).
    """
    
    # Count cards seen
    my_cards_of_rank = 0
    for i in range(len(hole_ranks)):
        if hole_ranks[i] == card_rank:
            my_cards_of_rank += 1
    
    board_cards_of_rank = 0
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
            if board_ranks[i] == card_rank:
                board_cards_of_rank += 1
    
    cards_of_rank_remaining = 4 - my_cards_of_rank - board_cards_of_rank
    if cards_of_rank_remaining == 0:
        return 0.0
    
    total_cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - total_cards_seen
    opponent_cards_total = num_opponents * 2
    board_cards_remaining = 5 - actual_board_cards
    
    # SCENARIO 1: Pocket pairs (2 of target rank in hole cards)
    prob_no_pocket_pairs = 1.0
    if cards_of_rank_remaining >= 2:
        # Ways to deal pocket pair to one opponent
        ways_pocket_pair = cards_of_rank_remaining * (cards_of_rank_remaining - 1) // 2
        total_ways_2_cards = remaining_deck_size * (remaining_deck_size - 1) // 2
        
        prob_one_opponent_pocket_pair = float(ways_pocket_pair) / float(total_ways_2_cards)
        prob_no_pocket_pairs = (1.0 - prob_one_opponent_pocket_pair) ** num_opponents
    
    # SCENARIO 2: One card + current board (if board has target rank)
    prob_no_current_single_pairs = 1.0
    if board_cards_of_rank > 0:
        # Probability one opponent has 0 cards of target rank
        ways_no_target_cards = (remaining_deck_size - cards_of_rank_remaining) * (remaining_deck_size - cards_of_rank_remaining - 1) // 2
        total_ways_2_cards = remaining_deck_size * (remaining_deck_size - 1) // 2
        
        prob_one_opponent_no_cards = float(ways_no_target_cards) / float(total_ways_2_cards)
        prob_no_current_single_pairs = prob_one_opponent_no_cards ** num_opponents
    
    # SCENARIO 3: One card + future board
    prob_no_future_single_pairs = 1.0
    if board_cards_remaining > 0 and board_cards_of_rank == 0:
        # Cards available for future board (after dealing to opponents)
        future_deck_size = remaining_deck_size - opponent_cards_total
        
        if future_deck_size > 0:
            # Probability future board contains target rank
            prob_future_board_no_target = 1.0
            for i in range(board_cards_remaining):
                prob_future_board_no_target *= float(future_deck_size - cards_of_rank_remaining - i) / float(future_deck_size - i)
            
            prob_future_board_has_target = 1.0 - prob_future_board_no_target
            
            # Probability at least one opponent has exactly 1 card of target rank
            # P(exactly 1) = 2 * (target_cards * non_target_cards) / (total * (total-1))
            prob_opponent_exactly_one = (
                2.0 * float(cards_of_rank_remaining) * float(remaining_deck_size - cards_of_rank_remaining) /
                (float(remaining_deck_size) * float(remaining_deck_size - 1))
            )
            
            prob_no_opponent_exactly_one = (1.0 - prob_opponent_exactly_one) ** num_opponents
            prob_at_least_one_opponent_exactly_one = 1.0 - prob_no_opponent_exactly_one
            
            # Combined probability for future scenario
            prob_future_pair = prob_future_board_has_target * prob_at_least_one_opponent_exactly_one
            prob_no_future_single_pairs = 1.0 - prob_future_pair
    
    # COMBINE ALL SCENARIOS
    # Using inclusion-exclusion approximation (assumes low overlap)
    prob_no_pairs_total = prob_no_pocket_pairs * prob_no_current_single_pairs * prob_no_future_single_pairs
    return 1.0 - prob_no_pairs_total



@njit
def calculate_trips_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    card_rank: int
) -> float:
    """
    Calculate probability of making three of a kind when remaining board cards are dealt.
    
    Args:
        hole_ranks, hole_suits: Your hole cards
        board_ranks, board_suits: Current board (padded)
        card_rank: Target rank (e.g., 14 for trip aces)
    
    Returns:
        Probability (0.0 to 1.0) of making trips
    """
    # Count cards of target rank
    my_cards_of_rank = 0
    for i in range(len(hole_ranks)):
        if hole_ranks[i] == card_rank:
            my_cards_of_rank += 1
    
    board_cards_of_rank = 0
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
            if board_ranks[i] == card_rank:
                board_cards_of_rank += 1
    
    total_cards_of_rank = my_cards_of_rank + board_cards_of_rank
    
    # Check if already have trips or better
    if total_cards_of_rank >= 3:
        return 1.0
    
    # Need at least 1 card to make trips possible
    if my_cards_of_rank == 0 and board_cards_of_rank == 0:
        return 0.0
    
    cards_of_rank_remaining = 4 - total_cards_of_rank
    if cards_of_rank_remaining == 0:
        return 0.0
    
    cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - cards_seen
    board_cards_remaining = 5 - actual_board_cards
    
    if board_cards_remaining == 0:
        return 0.0
    
    # Calculate based on how many we need
    cards_needed = 3 - total_cards_of_rank
    
    if cards_needed > board_cards_remaining or cards_needed > cards_of_rank_remaining:
        return 0.0
    
    # P(at least 'cards_needed' of target rank) = 1 - P(fewer than 'cards_needed')
    prob_not_enough = 0.0
    
    # Sum P(exactly 0) + P(exactly 1) + ... + P(exactly cards_needed-1)
    for got in range(cards_needed):
        if got > cards_of_rank_remaining:
            break
        
        # P(exactly 'got' cards of target rank in 'board_cards_remaining' draws)
        # = C(cards_of_rank_remaining, got) * C(non_target_remaining, board_cards_remaining-got) / C(remaining_deck_size, board_cards_remaining)
        
        non_target_remaining = remaining_deck_size - cards_of_rank_remaining
        
        # Calculate combinations
        ways_target = 1
        for i in range(got):
            ways_target = ways_target * (cards_of_rank_remaining - i) // (i + 1)
        
        ways_non_target = 1
        need_non_target = board_cards_remaining - got
        if need_non_target > non_target_remaining:
            continue  # Impossible
        for i in range(need_non_target):
            ways_non_target = ways_non_target * (non_target_remaining - i) // (i + 1)
        
        total_ways = 1
        for i in range(board_cards_remaining):
            total_ways = total_ways * (remaining_deck_size - i) // (i + 1)
        
        if total_ways > 0:
            prob_exactly_got = float(ways_target * ways_non_target) / float(total_ways)
            prob_not_enough += prob_exactly_got
    
    return 1.0 - prob_not_enough


@njit
def calculate_quads_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    card_rank: int
) -> float:
    """
    Calculate probability of making four of a kind when remaining board cards are dealt.
    """
    # Count cards of target rank
    my_cards_of_rank = 0
    for i in range(len(hole_ranks)):
        if hole_ranks[i] == card_rank:
            my_cards_of_rank += 1
    
    board_cards_of_rank = 0
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
            if board_ranks[i] == card_rank:
                board_cards_of_rank += 1
    
    total_cards_of_rank = my_cards_of_rank + board_cards_of_rank
    
    # Check if already have quads
    if total_cards_of_rank >= 4:
        return 1.0
    
    # Need at least 1 card to make quads possible
    if total_cards_of_rank == 0:
        return 0.0
    
    cards_of_rank_remaining = 4 - total_cards_of_rank
    cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - cards_seen
    board_cards_remaining = 5 - actual_board_cards
    
    if board_cards_remaining == 0 or cards_of_rank_remaining == 0:
        return 0.0
    
    cards_needed = 4 - total_cards_of_rank
    
    if cards_needed > board_cards_remaining or cards_needed > cards_of_rank_remaining:
        return 0.0
    
    # For quads, we need exactly 'cards_needed' more cards of the target rank
    # P(exactly 'cards_needed' of target rank)
    non_target_remaining = remaining_deck_size - cards_of_rank_remaining
    need_non_target = board_cards_remaining - cards_needed
    
    if need_non_target < 0 or need_non_target > non_target_remaining:
        return 0.0
    
    # C(cards_of_rank_remaining, cards_needed) * C(non_target_remaining, need_non_target) / C(remaining_deck_size, board_cards_remaining)
    ways_target = 1
    for i in range(cards_needed):
        ways_target = ways_target * (cards_of_rank_remaining - i) // (i + 1)
    
    ways_non_target = 1
    for i in range(need_non_target):
        ways_non_target = ways_non_target * (non_target_remaining - i) // (i + 1)
    
    total_ways = 1
    for i in range(board_cards_remaining):
        total_ways = total_ways * (remaining_deck_size - i) // (i + 1)
    
    if total_ways == 0:
        return 0.0
    
    return float(ways_target * ways_non_target) / float(total_ways)


# ============================================================================
# OPPONENT PROBABILITY FUNCTIONS (for opponents to make trips/quads)
# ============================================================================

@njit
def calculate_opponent_trips_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    card_rank: int,
    num_opponents: int
) -> float:
    """
    Calculate probability that at least one opponent will have trips by end of hand.
    
    Scenarios:
    1. Opponent has pocket pair + board gets one more
    2. Opponent has one card + board already has two  
    3. Opponent has one card + board gets two more
    4. Opponent has two cards + board gets one more
    """
    # Count cards seen
    my_cards_of_rank = 0
    for i in range(len(hole_ranks)):
        if hole_ranks[i] == card_rank:
            my_cards_of_rank += 1
    
    board_cards_of_rank = 0
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
            if board_ranks[i] == card_rank:
                board_cards_of_rank += 1
    
    cards_of_rank_remaining = 4 - my_cards_of_rank - board_cards_of_rank
    if cards_of_rank_remaining == 0:
        return 0.0
    
    total_cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - total_cards_seen
    opponent_cards_total = num_opponents * 2
    board_cards_remaining = 5 - actual_board_cards
    
    # SCENARIO 1: Opponent has pocket pair (2 cards) + needs 1 more from board
    prob_no_pocket_pair_trips = 1.0
    if cards_of_rank_remaining >= 2:
        # Probability one opponent has pocket pair
        prob_one_pocket_pair = (
            float(cards_of_rank_remaining) * float(cards_of_rank_remaining - 1) /
            (float(remaining_deck_size) * float(remaining_deck_size - 1))
        )
        prob_at_least_one_pocket_pair = 1.0 - (1.0 - prob_one_pocket_pair) ** num_opponents
        
        # If pocket pair exists, probability board gets at least 1 more
        if board_cards_of_rank >= 1:
            # Board already has one, so pocket pair = trips
            prob_board_completes_pocket = 1.0
        elif board_cards_remaining > 0:
            # Need board to get at least 1 more
            future_deck_size = remaining_deck_size - opponent_cards_total
            remaining_after_pocket = cards_of_rank_remaining - 2
            
            prob_board_gets_none = 1.0
            if future_deck_size > 0:
                for i in range(board_cards_remaining):
                    prob_board_gets_none *= float(future_deck_size - remaining_after_pocket - i) / float(future_deck_size - i)
            prob_board_completes_pocket = 1.0 - prob_board_gets_none
        else:
            prob_board_completes_pocket = 0.0
        
        prob_pocket_pair_makes_trips = prob_at_least_one_pocket_pair * prob_board_completes_pocket
        prob_no_pocket_pair_trips = 1.0 - prob_pocket_pair_makes_trips
    
    # SCENARIO 2: Opponent has 1 card + board has/gets 2+ more
    prob_no_single_card_trips = 1.0
    
    # Current board trips (board already has 2+)
    if board_cards_of_rank >= 2:
        # Any opponent with 1 card has trips
        prob_one_opponent_no_cards = (
            float(remaining_deck_size - cards_of_rank_remaining) *
            float(remaining_deck_size - cards_of_rank_remaining - 1) /
            (float(remaining_deck_size) * float(remaining_deck_size - 1))
        )
        prob_no_single_card_trips = prob_one_opponent_no_cards ** num_opponents
    
    # Future board trips (if board currently has 0 or 1, could get to 2+)
    elif board_cards_remaining >= (2 - board_cards_of_rank):
        # Simplified: if opponent has 1 card, what's probability board gets enough?
        prob_opponent_has_exactly_one = (
            2.0 * float(cards_of_rank_remaining) * float(remaining_deck_size - cards_of_rank_remaining) /
            (float(remaining_deck_size) * float(remaining_deck_size - 1))
        )
        prob_at_least_one_has_one = 1.0 - (1.0 - prob_opponent_has_exactly_one) ** num_opponents
        
        # Probability board gets enough cards (simplified)
        need_from_board = 2 - board_cards_of_rank
        future_deck_size = remaining_deck_size - opponent_cards_total
        
        if future_deck_size > 0 and need_from_board <= board_cards_remaining:
            # Very rough approximation for board getting exactly what we need
            prob_board_gets_enough = min(1.0, float(need_from_board * cards_of_rank_remaining) / float(future_deck_size))
        else:
            prob_board_gets_enough = 0.0
        
        prob_future_single_trips = prob_at_least_one_has_one * prob_board_gets_enough
        prob_no_single_card_trips = min(prob_no_single_card_trips, 1.0 - prob_future_single_trips)
    
    # Combine scenarios (independence approximation)
    prob_no_trips = prob_no_pocket_pair_trips * prob_no_single_card_trips
    
    return 1.0 - prob_no_trips


@njit
def calculate_opponent_quads_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    card_rank: int,
    num_opponents: int
) -> float:
    """
    Calculate probability that at least one opponent will have quads by end of hand.
    
    Main scenarios:
    1. Opponent has pocket pair + board gets two more
    2. Opponent has one card + board gets three more (rare)
    """
    # Count cards seen
    my_cards_of_rank = 0
    for i in range(len(hole_ranks)):
        if hole_ranks[i] == card_rank:
            my_cards_of_rank += 1
    
    board_cards_of_rank = 0
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
            if board_ranks[i] == card_rank:
                board_cards_of_rank += 1
    
    cards_of_rank_remaining = 4 - my_cards_of_rank - board_cards_of_rank
    if cards_of_rank_remaining < 2:  # Need at least 2 for any quads scenario
        return 0.0
    
    total_cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - total_cards_seen
    opponent_cards_total = num_opponents * 2
    board_cards_remaining = 5 - actual_board_cards
    
    # SCENARIO 1: Opponent has pocket pair + board gets 2 more
    prob_no_pocket_quads = 1.0
    if cards_of_rank_remaining >= 2:
        # Probability at least one opponent has pocket pair
        prob_one_pocket_pair = (
            float(cards_of_rank_remaining) * float(cards_of_rank_remaining - 1) /
            (float(remaining_deck_size) * float(remaining_deck_size - 1))
        )
        prob_at_least_one_pocket_pair = 1.0 - (1.0 - prob_one_pocket_pair) ** num_opponents
        
        # Probability board gets exactly 2 more (for quads)
        need_from_board = 2 - board_cards_of_rank
        if need_from_board <= 0:
            # Board already has enough, pocket pair makes quads
            prob_board_completes = 1.0
        elif need_from_board <= board_cards_remaining and need_from_board <= (cards_of_rank_remaining - 2):
            # Calculate probability board gets exactly what we need
            future_deck_size = remaining_deck_size - opponent_cards_total
            remaining_target_cards = cards_of_rank_remaining - 2  # After pocket pair
            
            if future_deck_size > 0:
                # Combination calculation for exactly 'need_from_board' cards
                ways_target = 1
                for i in range(need_from_board):
                    ways_target = ways_target * (remaining_target_cards - i) // (i + 1)
                
                ways_non_target = 1
                need_non_target = board_cards_remaining - need_from_board
                non_target_available = future_deck_size - remaining_target_cards
                for i in range(need_non_target):
                    ways_non_target = ways_non_target * (non_target_available - i) // (i + 1)
                
                total_ways = 1
                for i in range(board_cards_remaining):
                    total_ways = total_ways * (future_deck_size - i) // (i + 1)
                
                prob_board_completes = float(ways_target * ways_non_target) / float(total_ways) if total_ways > 0 else 0.0
            else:
                prob_board_completes = 0.0
        else:
            prob_board_completes = 0.0
        
        prob_pocket_quads = prob_at_least_one_pocket_pair * prob_board_completes
        prob_no_pocket_quads = 1.0 - prob_pocket_quads
    
    # SCENARIO 2: Single card + board gets 3 (very rare, simplified)
    prob_no_single_quads = 1.0
    if board_cards_of_rank == 0 and board_cards_remaining >= 3:
        # Very rough approximation - usually negligible
        prob_opponent_has_one = (
            2.0 * float(cards_of_rank_remaining) * float(remaining_deck_size - cards_of_rank_remaining) /
            (float(remaining_deck_size) * float(remaining_deck_size - 1))
        )
        # Probability is very small, so we'll approximate as 0 for efficiency
        prob_no_single_quads = 1.0
    
    return 1.0 - (prob_no_pocket_quads * prob_no_single_quads)



@njit
def encode_single_card_with_presence(rank: int, suit: int) -> float:
    """
    Missing cards = 0.0
    Present cards = positive values based on rank/suit
    """
    if rank <= 1 or suit < 0:  # Missing/invalid card
        return 0.0
    
    # Encode present cards as positive values
    rank_component = float(rank) / 14.0  # 0.14 to 1.0
    suit_component = float(suit) / 40.0  # Small component: 0.0 to 0.075
    
    return rank_component + suit_component  # Always > 0 for present cards


# Your final function:
@njit
def create_dense_card_representation(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray
) -> np.ndarray:
    features = np.zeros(7, dtype=np.float32)
    
    # Hole cards (always present)
    for i in range(2):
        features[i] = encode_single_card_with_presence(hole_ranks[i], hole_suits[i])
    
    # Board cards (may be missing)
    for i in range(5):
        if i < len(board_ranks) and board_ranks[i] > 1:
            features[i + 2] = encode_single_card_with_presence(board_ranks[i], board_suits[i])
        else:
            features[i + 2] = 0.0  # Missing
    
    return features



@njit
def calculate_fullhouse_probability(
   hole_ranks: np.ndarray,
   hole_suits: np.ndarray, 
   board_ranks: np.ndarray,
   board_suits: np.ndarray
) -> float:
   """
   FIXED: Efficient full house calculation using pure combinatorics.
   """
   
   # Quick analysis of current hand state
   rank_counts = np.zeros(15, dtype=np.int32)
   
   for i in range(len(hole_ranks)):
       if hole_ranks[i] > 1:
           rank_counts[hole_ranks[i]] += 1
   
   actual_board_cards = 0
   for i in range(len(board_ranks)):
       if board_ranks[i] > 1:
           actual_board_cards += 1
           rank_counts[board_ranks[i]] += 1
   
   # Categorize hand state
   trips_count = 0
   pair_count = 0
   
   for rank in range(2, 15):
       count = rank_counts[rank]
       if count >= 3:
           trips_count += 1
       elif count == 2:
           pair_count += 1
   
   # Early returns for definitive cases
   if trips_count >= 1 and pair_count >= 1:
       return 1.0  # Already have full house
   
   remaining_deck_size = 52 - len(hole_ranks) - actual_board_cards
   board_cards_remaining = 5 - actual_board_cards
   
   if board_cards_remaining == 0:
       return 0.0
   
   # CASE 1: Have trips, need any pair
   if trips_count == 1 and pair_count == 0:
       if board_cards_remaining < 2:
           return 0.0  # Can't make pair with <2 cards
       
       # Calculate total ways to get pairs from available ranks
       total_pair_ways = 0
       for rank in range(2, 15):
           available_of_rank = 4 - rank_counts[rank]
           if rank_counts[rank] < 3 and available_of_rank >= 2:  # Not trips rank, can pair
               pair_ways = available_of_rank * (available_of_rank - 1) // 2
               total_pair_ways += pair_ways
       
       total_ways = remaining_deck_size * (remaining_deck_size - 1) // 2
       return float(total_pair_ways) / float(total_ways)
   
   # CASE 2: Two pair, need one more of either - FIXED ORDER
   elif pair_count >= 2:
       total_helpful_cards = 0
       for rank in range(2, 15):
           if rank_counts[rank] == 2:
               total_helpful_cards += 2  # 2 cards left of each pair rank
       
       # P(at least one helpful card) - FIXED FORMULA
       prob_miss_all = 1.0
       for i in range(board_cards_remaining):
           prob_miss_all *= float(remaining_deck_size - total_helpful_cards - i) / float(remaining_deck_size - i)
       
       return 1.0 - prob_miss_all
   
   # CASE 3: Have single pair, need trips of same rank
   elif pair_count == 1 and trips_count == 0:
       # Find pair rank
       pair_rank = 0
       for rank in range(2, 15):
           if rank_counts[rank] == 2:
               pair_rank = rank
               break
       
       available_of_pair_rank = 2  # 2 cards left of pair rank
       
       # P(at least 1 more of pair rank)
       prob_miss_all = 1.0
       for i in range(board_cards_remaining):
           prob_miss_all *= float(remaining_deck_size - available_of_pair_rank - i) / float(remaining_deck_size - i)
       
       return 1.0 - prob_miss_all
   
   return 0.0


@njit
def calculate_opponent_fullhouse_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    num_opponents: int
) -> float:
    """
    FIXED: Efficient opponent full house calculation.
    """
    
    # Analyze board state
    board_rank_counts = np.zeros(15, dtype=np.int32)
    actual_board_cards = 0
    
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
            board_rank_counts[board_ranks[i]] += 1
    
    # Count your cards
    my_rank_counts = np.zeros(15, dtype=np.int32)
    for i in range(len(hole_ranks)):
        if hole_ranks[i] > 1:
            my_rank_counts[hole_ranks[i]] += 1
    
    # Find board state
    board_pair_rank = -1
    for rank in range(2, 15):
        if board_rank_counts[rank] == 2:
            board_pair_rank = rank
            break
    
    remaining_deck_size = 52 - len(hole_ranks) - actual_board_cards
    
    # MAIN CASE: Board has pair - opponent pocket pair of different rank = instant full house
    if board_pair_rank >= 0:
        total_pocket_pair_ways = 0
        
        for rank in range(2, 15):
            if rank != board_pair_rank:  # Different from board pair
                available_of_rank = 4 - my_rank_counts[rank] - board_rank_counts[rank]
                if available_of_rank >= 2:
                    pocket_ways = available_of_rank * (available_of_rank - 1) // 2
                    total_pocket_pair_ways += pocket_ways
        
        total_2card_ways = remaining_deck_size * (remaining_deck_size - 1) // 2
        if total_2card_ways > 0:
            prob_one_opponent_pocket = float(total_pocket_pair_ways) / float(total_2card_ways)
            prob_at_least_one_pocket = 1.0 - (1.0 - prob_one_opponent_pocket) ** num_opponents
            return prob_at_least_one_pocket
    
    # Simplified for other cases
    return 0.0


@njit
def calculate_flush_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray
) -> float:
    """
    Calculate probability of making a flush when remaining board cards are dealt.
    Uses exact combinatorial mathematics for precision.
    
    Returns probability (0.0 to 1.0) of having flush by river.
    """
    # Count actual board cards (skip padding)
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
    
    board_cards_remaining = 5 - actual_board_cards
    
    # For each suit, calculate if we can make a flush
    max_flush_prob = 0.0
    
    for target_suit in range(4):
        # Count cards of this suit we have
        my_suit_count = 0
        board_suit_count = 0
        
        # Count hole cards of target suit
        for i in range(len(hole_suits)):
            if hole_suits[i] == target_suit and hole_ranks[i] > 1:
                my_suit_count += 1
        
        # Count board cards of target suit
        for i in range(len(board_suits)):
            if i < len(board_suits) and board_suits[i] == target_suit and board_ranks[i] > 1:
                board_suit_count += 1
        
        total_suit_count = my_suit_count + board_suit_count
        
        # Check if already have flush
        if total_suit_count >= 5:
            return 1.0
        
        # Skip if no potential (need at least 3 by river to make flush possible)
        max_possible = total_suit_count + board_cards_remaining
        if max_possible < 5:
            continue
        
        # Calculate cards of this suit remaining in deck
        cards_seen = len(hole_ranks) + actual_board_cards
        remaining_deck_size = 52 - cards_seen
        cards_of_suit_remaining = 13 - total_suit_count
        cards_needed = 5 - total_suit_count
        
        if cards_needed <= 0:
            return 1.0  # Already have flush
        
        if cards_needed > board_cards_remaining or cards_needed > cards_of_suit_remaining:
            continue  # Impossible
        
        # Calculate probability using hypergeometric distribution
        # P(at least cards_needed of target suit) = 1 - P(fewer than cards_needed)
        prob_not_enough = 0.0
        
        # Sum P(exactly 0) + P(exactly 1) + ... + P(exactly cards_needed-1)
        for got in range(cards_needed):
            if got > cards_of_suit_remaining:
                break
            
            other_cards_remaining = remaining_deck_size - cards_of_suit_remaining
            need_other_cards = board_cards_remaining - got
            
            if need_other_cards < 0 or need_other_cards > other_cards_remaining:
                continue
            
            # Calculate combinations
            ways_target = 1
            for i in range(got):
                ways_target = ways_target * (cards_of_suit_remaining - i) // (i + 1)
            
            ways_other = 1
            for i in range(need_other_cards):
                ways_other = ways_other * (other_cards_remaining - i) // (i + 1)
            
            total_ways = 1
            for i in range(board_cards_remaining):
                total_ways = total_ways * (remaining_deck_size - i) // (i + 1)
            
            if total_ways > 0:
                prob_exactly_got = float(ways_target * ways_other) / float(total_ways)
                prob_not_enough += prob_exactly_got
        
        flush_prob = 1.0 - prob_not_enough
        max_flush_prob = max(max_flush_prob, flush_prob)
    
    return max_flush_prob


@njit
def calculate_opponent_flush_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    num_opponents: int
) -> float:
    """
    Calculate probability that at least one opponent will have a flush by river.
    Conservative approach focusing on most likely scenarios.
    """
    # Count actual board cards
    actual_board_cards = 0
    board_suit_counts = np.zeros(4, dtype=np.int32)
    
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
            if board_suits[i] >= 0:
                board_suit_counts[board_suits[i]] += 1
    
    # CRITICAL FIX: Check if board already has 5+ of any suit (everyone has flush)
    for suit in range(4):
        if board_suit_counts[suit] >= 5:
            return 1.0
    
    # Count my cards by suit
    my_suit_counts = np.zeros(4, dtype=np.int32)
    for i in range(len(hole_suits)):
        if hole_suits[i] >= 0 and hole_ranks[i] > 1:
            my_suit_counts[hole_suits[i]] += 1
    
    cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - cards_seen
    board_cards_remaining = 5 - actual_board_cards
    
    if remaining_deck_size <= 0 or num_opponents <= 0:
        return 0.0
    
    # Find the suit with the highest flush potential for opponents
    max_opponent_flush_prob = 0.0
    
    for suit in range(4):
        total_suit_cards_seen = my_suit_counts[suit] + board_suit_counts[suit]
        suit_cards_remaining = 13 - total_suit_cards_seen
        
        if suit_cards_remaining < 2:  # Need at least 2 for any flush scenario
            continue
        
        # Main scenario: Suited hole cards + board completes
        suit_flush_prob = 0.0
        
        # Probability at least one opponent has suited hole cards of this suit
        prob_one_suited = (
            float(suit_cards_remaining) * float(suit_cards_remaining - 1) /
            (float(remaining_deck_size) * float(remaining_deck_size - 1))
        )
        prob_at_least_one_suited = 1.0 - (1.0 - prob_one_suited) ** num_opponents
        
        # Given suited hole cards, probability board completes flush
        total_with_suited = board_suit_counts[suit] + 2
        need_from_board = max(0, 5 - total_with_suited)
        
        if need_from_board <= 0:
            # Already flush with current board + suited hole cards
            prob_board_completes = 1.0
        elif need_from_board <= board_cards_remaining:
            # Board needs to provide more cards
            future_deck_size = remaining_deck_size - (num_opponents * 2)
            future_suit_cards = suit_cards_remaining - 2  # After suited hole cards
            
            if future_deck_size > 0 and future_suit_cards >= need_from_board:
                # FIXED: Use exact hypergeometric probability
                if need_from_board == 1:
                    # Need at least 1 more card of this suit from remaining board cards
                    prob_no_target_cards = 1.0
                    other_cards = future_deck_size - future_suit_cards
                    for i in range(board_cards_remaining):
                        prob_no_target_cards *= float(other_cards - i) / float(future_deck_size - i)
                    prob_board_completes = 1.0 - prob_no_target_cards
                elif need_from_board == 2:
                    # Need exactly 2 more cards of this suit
                    if future_suit_cards >= 2 and future_deck_size >= 2:
                        prob_board_completes = (
                            float(future_suit_cards) * float(future_suit_cards - 1) /
                            (float(future_deck_size) * float(future_deck_size - 1))
                        )
                    else:
                        prob_board_completes = 0.0
                else:
                    # Need 3+ cards - use hypergeometric but cap at reasonable value
                    prob_board_completes = 0.0
                    if need_from_board <= board_cards_remaining and future_suit_cards >= need_from_board:
                        # Calculate exact hypergeometric probability
                        ways_success = 1
                        for i in range(need_from_board):
                            ways_success = ways_success * (future_suit_cards - i) // (i + 1)
                        
                        ways_failure = 1
                        need_other = board_cards_remaining - need_from_board
                        other_cards = future_deck_size - future_suit_cards
                        for i in range(need_other):
                            ways_failure = ways_failure * (other_cards - i) // (i + 1)
                        
                        total_ways = 1
                        for i in range(board_cards_remaining):
                            total_ways = total_ways * (future_deck_size - i) // (i + 1)
                        
                        if total_ways > 0:
                            prob_board_completes = float(ways_success * ways_failure) / float(total_ways)
                        else:
                            prob_board_completes = 0.0
                    else:
                        prob_board_completes = 0.0
            else:
                prob_board_completes = 0.0
        else:
            prob_board_completes = 0.0
        
        suit_flush_prob = prob_at_least_one_suited * prob_board_completes
        
        # Also consider single card + 4-card board scenario (but more conservatively)
        if board_suit_counts[suit] + board_cards_remaining >= 4 and board_suit_counts[suit] >= 2:
            # Only consider if board already has 2+ of this suit
            need_board_total = 4
            need_from_future = max(0, need_board_total - board_suit_counts[suit])
            
            if need_from_future <= board_cards_remaining and suit_cards_remaining >= 1:
                # Probability one opponent has exactly 1 of this suit
                prob_one_card = (
                    2.0 * float(suit_cards_remaining) * float(remaining_deck_size - suit_cards_remaining) /
                    (float(remaining_deck_size) * float(remaining_deck_size - 1))
                )
                prob_someone_has_one = 1.0 - (1.0 - prob_one_card) ** num_opponents
                
                # Conservative estimate: board gets exactly what's needed
                if need_from_future == 0:
                    prob_board_cooperates = 1.0
                elif need_from_future == 1:
                    future_deck_size = remaining_deck_size - (num_opponents * 2)
                    future_suit_cards = suit_cards_remaining - 1
                    if future_deck_size > 0:
                        prob_board_cooperates = float(future_suit_cards) / float(future_deck_size)
                    else:
                        prob_board_cooperates = 0.0
                elif need_from_future == 2:
                    future_deck_size = remaining_deck_size - (num_opponents * 2)
                    future_suit_cards = suit_cards_remaining - 1
                    if future_deck_size >= 2 and future_suit_cards >= 2:
                        prob_board_cooperates = (
                            float(future_suit_cards) * float(future_suit_cards - 1) /
                            (float(future_deck_size) * float(future_deck_size - 1))
                        )
                    else:
                        prob_board_cooperates = 0.0
                else:
                    prob_board_cooperates = 0.0
                
                single_card_prob = prob_someone_has_one * prob_board_cooperates
                
                # Add to suit probability (capped to avoid double-counting)
                suit_flush_prob = min(1.0, suit_flush_prob + single_card_prob * 0.3)  # Further discount
        
        max_opponent_flush_prob = max(max_opponent_flush_prob, suit_flush_prob)
    
    return max_opponent_flush_prob



@njit
def DEPRECATED_calculate_straight_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray
) -> float:
    """
    Calculate probability of making a straight when remaining board cards are dealt.
    Uses exact combinatorial mathematics for precision.
    
    Returns probability (0.0 to 1.0) of having straight by river.
    """
    # Count actual board cards (skip padding)
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
    
    board_cards_remaining = 5 - actual_board_cards
    
    # Count rank occurrences
    rank_counts = np.zeros(15, dtype=np.int32)
    
    # Count hole cards
    for r in hole_ranks:
        if r > 1:
            rank_counts[r] += 1
    
    # Count board cards
    for i in range(len(board_ranks)):
        if i < len(board_ranks) and board_ranks[i] > 1:
            rank_counts[board_ranks[i]] += 1
    
    # Create rank presence flags
    rank_present = np.zeros(15, dtype=np.uint8)
    for rank in range(2, 15):
        if rank_counts[rank] > 0:
            rank_present[rank] = 1
    
    # Check if already have straight
    # Check normal straights (2-3-4-5-6 through T-J-Q-K-A)
    for low in range(2, 11):
        consecutive_count = 0
        for i in range(5):
            if rank_present[low + i] == 1:
                consecutive_count += 1
        if consecutive_count == 5:
            return 1.0
    
    # Check wheel (A-2-3-4-5)
    wheel_count = 0
    if rank_present[14] == 1:  # Ace
        wheel_count += 1
    if rank_present[2] == 1:   # 2
        wheel_count += 1
    if rank_present[3] == 1:   # 3
        wheel_count += 1
    if rank_present[4] == 1:   # 4
        wheel_count += 1
    if rank_present[5] == 1:   # 5
        wheel_count += 1
    if wheel_count == 5:
        return 1.0
    
    if board_cards_remaining == 0:
        return 0.0
    
    # Calculate remaining deck size
    cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - cards_seen
    
    # Find best straight draw probability
    max_straight_prob = 0.0
    
    # Define all possible straights as arrays
    straight_definitions = np.array([
        [14, 2, 3, 4, 5],    # A-2-3-4-5 (wheel)
        [2, 3, 4, 5, 6],     # 2-3-4-5-6
        [3, 4, 5, 6, 7],     # 3-4-5-6-7
        [4, 5, 6, 7, 8],     # 4-5-6-7-8
        [5, 6, 7, 8, 9],     # 5-6-7-8-9
        [6, 7, 8, 9, 10],    # 6-7-8-9-T
        [7, 8, 9, 10, 11],   # 7-8-9-T-J
        [8, 9, 10, 11, 12],  # 8-9-T-J-Q
        [9, 10, 11, 12, 13], # 9-T-J-Q-K
        [10, 11, 12, 13, 14] # T-J-Q-K-A
    ], dtype=np.int32)
    
    # Find the best straight draw probability more carefully
    
    # Check for immediate 1-card completions (open-ended, gutshot)
    immediate_outs = np.zeros(15, dtype=np.int32)
    
    for straight_idx in range(10):
        straight_ranks = straight_definitions[straight_idx]
        
        # Count what we need for this straight
        need_count = 0
        need_rank = 0
        
        for i in range(5):
            rank = straight_ranks[i]
            if rank_present[rank] == 0:
                need_count += 1
                need_rank = rank
        
        # If we need exactly 1 card for a straight
        if need_count == 1:
            outs = 4 - rank_counts[need_rank]
            if outs > 0:
                immediate_outs[need_rank] = max(immediate_outs[need_rank], outs)
    
    # Calculate probability for immediate completions
    total_immediate_outs = 0
    for rank in range(2, 15):
        if immediate_outs[rank] > 0:
            total_immediate_outs += immediate_outs[rank]
    
    if total_immediate_outs > 0:
        prob_miss_all = 1.0
        non_outs = remaining_deck_size - total_immediate_outs
        for i in range(board_cards_remaining):
            prob_miss_all *= float(non_outs - i) / float(remaining_deck_size - i)
        max_straight_prob = 1.0 - prob_miss_all
    
    # Only check 2-card scenarios if no immediate completions and we have enough cards
    elif board_cards_remaining >= 2:
        best_two_card_prob = 0.0
        
        for straight_idx in range(10):
            straight_ranks = straight_definitions[straight_idx]
            
            # Count what we need for this straight
            need_ranks = np.zeros(5, dtype=np.int32)
            need_count = 0
            
            for i in range(5):
                rank = straight_ranks[i]
                if rank_present[rank] == 0:
                    if need_count < 5:
                        need_ranks[need_count] = rank
                    need_count += 1
            
            # Only consider 2-card draws that are realistic
            if need_count == 2:
                rank1 = need_ranks[0]
                rank2 = need_ranks[1]
                outs1 = 4 - rank_counts[rank1]
                outs2 = 4 - rank_counts[rank2]
                
                if outs1 > 0 and outs2 > 0:
                    if board_cards_remaining == 2:
                        # Need both specific ranks in exactly 2 cards
                        straight_prob = float(outs1 * outs2) / float(remaining_deck_size * (remaining_deck_size - 1))
                    else:
                        # Use inclusion-exclusion for more cards
                        prob_miss_1 = 1.0
                        prob_miss_2 = 1.0
                        prob_miss_both = 1.0
                        
                        non_outs_1 = remaining_deck_size - outs1
                        non_outs_2 = remaining_deck_size - outs2
                        non_outs_both = remaining_deck_size - outs1 - outs2
                        
                        for i in range(board_cards_remaining):
                            prob_miss_1 *= float(non_outs_1 - i) / float(remaining_deck_size - i)
                            prob_miss_2 *= float(non_outs_2 - i) / float(remaining_deck_size - i)
                            prob_miss_both *= float(non_outs_both - i) / float(remaining_deck_size - i)
                        
                        straight_prob = 1.0 - prob_miss_1 - prob_miss_2 + prob_miss_both
                    
                    best_two_card_prob = max(best_two_card_prob, straight_prob)
            
            # Very conservative check for 3+ card needs (rare backdoor scenarios)
            elif need_count == 3 and board_cards_remaining >= 3:
                # Only if we have exactly what we need and it's reasonably likely
                if need_count == board_cards_remaining:
                    rank1 = need_ranks[0]
                    rank2 = need_ranks[1] 
                    rank3 = need_ranks[2]
                    outs1 = 4 - rank_counts[rank1]
                    outs2 = 4 - rank_counts[rank2]
                    outs3 = 4 - rank_counts[rank3]
                    
                    if outs1 > 0 and outs2 > 0 and outs3 > 0:
                        # Very conservative probability for 3-card scenario
                        straight_prob = float(outs1 * outs2 * outs3) / float(
                            remaining_deck_size * (remaining_deck_size - 1) * (remaining_deck_size - 2)
                        )
                        best_two_card_prob = max(best_two_card_prob, straight_prob)
        
        max_straight_prob = best_two_card_prob
    
    return max_straight_prob


@njit
def DEPRECATED_calculate_opponent_straight_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    num_opponents: int
) -> float:
    """
    Calculate probability that at least one opponent will have a straight by river.
    """
    # Count actual board cards
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
    
    # Count board rank occurrences
    board_rank_counts = np.zeros(15, dtype=np.int32)
    board_rank_present = np.zeros(15, dtype=np.uint8)
    
    for i in range(len(board_ranks)):
        if i < len(board_ranks) and board_ranks[i] > 1:
            rank = board_ranks[i]
            board_rank_counts[rank] += 1
            board_rank_present[rank] = 1
    
    # Check if board already has straight
    # Check normal straights
    for low in range(2, 11):
        consecutive_count = 0
        for i in range(5):
            if board_rank_present[low + i] == 1:
                consecutive_count += 1
        if consecutive_count >= 5:
            return 1.0
    
    # Check wheel on board
    wheel_count = 0
    if board_rank_present[14] == 1:  # Ace
        wheel_count += 1
    if board_rank_present[2] == 1:   # 2
        wheel_count += 1
    if board_rank_present[3] == 1:   # 3
        wheel_count += 1
    if board_rank_present[4] == 1:   # 4
        wheel_count += 1
    if board_rank_present[5] == 1:   # 5
        wheel_count += 1
    if wheel_count >= 5:
        return 1.0
    
    cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - cards_seen
    board_cards_remaining = 5 - actual_board_cards
    
    if remaining_deck_size <= 0 or num_opponents <= 0:
        return 0.0
    
    # Count my cards
    my_rank_counts = np.zeros(15, dtype=np.int32)
    for r in hole_ranks:
        if r > 1:
            my_rank_counts[r] += 1
    
    max_opponent_straight_prob = 0.0
    
    # Define straights
    straight_definitions = np.array([
        [14, 2, 3, 4, 5],    # A-2-3-4-5 (wheel)
        [2, 3, 4, 5, 6],     # 2-3-4-5-6
        [3, 4, 5, 6, 7],     # 3-4-5-6-7
        [4, 5, 6, 7, 8],     # 4-5-6-7-8
        [5, 6, 7, 8, 9],     # 5-6-7-8-9
        [6, 7, 8, 9, 10],    # 6-7-8-9-T
        [7, 8, 9, 10, 11],   # 7-8-9-T-J
        [8, 9, 10, 11, 12],  # 8-9-T-J-Q
        [9, 10, 11, 12, 13], # 9-T-J-Q-K
        [10, 11, 12, 13, 14] # T-J-Q-K-A
    ], dtype=np.int32)
    
    # Check each possible straight opponents could make
    for straight_idx in range(10):
        straight_ranks = straight_definitions[straight_idx]
        
        # Count how many ranks board has for this straight
        board_has = 0
        need_ranks = np.zeros(5, dtype=np.int32)
        need_count = 0
        
        for i in range(5):
            rank = straight_ranks[i]
            if board_rank_present[rank] == 1:
                board_has += 1
            else:
                need_ranks[need_count] = rank
                need_count += 1
        
        need_from_opponents = need_count
        
        if need_from_opponents > 2:  # Focus on realistic scenarios
            continue
        
        if need_from_opponents == 0:
            return 1.0  # Board already has straight
        
        straight_prob = 0.0
        
        if need_from_opponents == 1:
            # Opponents need exactly 1 rank
            need_rank = need_ranks[0]
            available = 4 - my_rank_counts[need_rank] - board_rank_counts[need_rank]
            
            if available > 0:
                # Probability at least one opponent has this rank
                prob_one_has = 2.0 * float(available) / float(remaining_deck_size)
                prob_at_least_one = 1.0 - (1.0 - prob_one_has) ** num_opponents
                straight_prob = prob_at_least_one
        
        elif need_from_opponents == 2:
            # Opponents need exactly 2 ranks - could have both in hole cards
            rank1 = need_ranks[0]
            rank2 = need_ranks[1]
            avail1 = 4 - my_rank_counts[rank1] - board_rank_counts[rank1]
            avail2 = 4 - my_rank_counts[rank2] - board_rank_counts[rank2]
            
            if avail1 > 0 and avail2 > 0:
                # Probability one opponent has both ranks in hole cards
                prob_both = float(avail1 * avail2) / float(remaining_deck_size * (remaining_deck_size - 1))
                prob_at_least_one_both = 1.0 - (1.0 - prob_both) ** num_opponents
                straight_prob = prob_at_least_one_both
        
        max_opponent_straight_prob = max(max_opponent_straight_prob, straight_prob)
    
    return max_opponent_straight_prob

@njit
def calculate_straight_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray
) -> float:
    """
    Calculate probability of making a straight when remaining board cards are dealt.
    Uses exact combinatorial mathematics for precision.
    
    Returns probability (0.0 to 1.0) of having straight by river.
    """
    # Count actual board cards (skip padding)
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
    
    board_cards_remaining = 5 - actual_board_cards
    
    # Count rank occurrences including all cards
    rank_counts = np.zeros(15, dtype=np.int32)
    
    # Count hole cards
    for r in hole_ranks:
        if r > 1:
            rank_counts[r] += 1
    
    # Count board cards
    for i in range(len(board_ranks)):
        if i < len(board_ranks) and board_ranks[i] > 1:
            rank_counts[board_ranks[i]] += 1
    
    # Create rank presence flags
    rank_present = np.zeros(15, dtype=np.uint8)
    for rank in range(2, 15):
        if rank_counts[rank] > 0:
            rank_present[rank] = 1
    
    # Check if already have straight
    # Check normal straights (2-3-4-5-6 through T-J-Q-K-A)
    for low in range(2, 11):
        consecutive_count = 0
        for i in range(5):
            if rank_present[low + i] == 1:
                consecutive_count += 1
        if consecutive_count == 5:
            return 1.0
    
    # Check wheel (A-2-3-4-5)
    wheel_count = 0
    if rank_present[14] == 1:  # Ace
        wheel_count += 1
    if rank_present[2] == 1:   # 2
        wheel_count += 1
    if rank_present[3] == 1:   # 3
        wheel_count += 1
    if rank_present[4] == 1:   # 4
        wheel_count += 1
    if rank_present[5] == 1:   # 5
        wheel_count += 1
    if wheel_count == 5:
        return 1.0
    
    if board_cards_remaining == 0:
        return 0.0
    
    # Calculate remaining deck size
    cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - cards_seen
    
    # Find all possible ways to complete straights, then eliminate supersets
    straight_definitions = np.array([
        [14, 2, 3, 4, 5],    # A-2-3-4-5 (wheel)
        [2, 3, 4, 5, 6],     # 2-3-4-5-6
        [3, 4, 5, 6, 7],     # 3-4-5-6-7
        [4, 5, 6, 7, 8],     # 4-5-6-7-8
        [5, 6, 7, 8, 9],     # 5-6-7-8-9
        [6, 7, 8, 9, 10],    # 6-7-8-9-T
        [7, 8, 9, 10, 11],   # 7-8-9-T-J
        [8, 9, 10, 11, 12],  # 8-9-T-J-Q
        [9, 10, 11, 12, 13], # 9-T-J-Q-K
        [10, 11, 12, 13, 14] # T-J-Q-K-A
    ], dtype=np.int32)
    
    # Collect all requirement sets (what cards we need for each straight)
    max_requirement_sets = 20  # Upper bound on number of requirement sets
    requirement_sets = np.zeros((max_requirement_sets, 5), dtype=np.int32)  # Each set can have up to 5 missing ranks
    requirement_sizes = np.zeros(max_requirement_sets, dtype=np.int32)
    num_requirement_sets = 0
    
    for straight_idx in range(10):
        straight_ranks = straight_definitions[straight_idx]
        
        # Find what we need for this straight
        missing_ranks = np.zeros(5, dtype=np.int32)
        missing_count = 0
        have_count = 0
        
        for i in range(5):
            rank = straight_ranks[i]
            if rank_present[rank] == 1:
                have_count += 1
            else:
                missing_ranks[missing_count] = rank
                missing_count += 1
        
        # Only consider viable draws (need reasonable number of cards and have enough board cards remaining)
        if missing_count > 0 and missing_count <= board_cards_remaining and missing_count <= 3:
            # Check that all needed ranks are available
            all_available = True
            for i in range(missing_count):
                if 4 - rank_counts[missing_ranks[i]] <= 0:
                    all_available = False
                    break
            
            if all_available:
                # Check connectivity for 2+ card requirements
                if missing_count >= 2:
                    our_cards = np.zeros(5, dtype=np.int32)
                    our_count = 0
                    for rank in straight_ranks:
                        if rank_present[rank] == 1:
                            our_cards[our_count] = rank
                            our_count += 1
                    
                    # Sort our cards
                    for i in range(our_count):
                        for j in range(i + 1, our_count):
                            if our_cards[i] > our_cards[j]:
                                temp = our_cards[i]
                                our_cards[i] = our_cards[j]
                                our_cards[j] = temp
                    
                    # Check connectivity
                    reasonable = True
                    if our_count >= 2:
                        for j in range(our_count - 1):
                            gap = our_cards[j+1] - our_cards[j] - 1
                            if gap > 1:
                                reasonable = False
                                break
                    
                    if not reasonable:
                        continue
                
                # Add this requirement set
                if num_requirement_sets < max_requirement_sets:
                    for i in range(missing_count):
                        requirement_sets[num_requirement_sets][i] = missing_ranks[i]
                    requirement_sizes[num_requirement_sets] = missing_count
                    num_requirement_sets += 1
    
    # Now eliminate supersets: if set A is a subset of set B, remove set B
    minimal_sets = np.zeros((max_requirement_sets, 5), dtype=np.int32)
    minimal_sizes = np.zeros(max_requirement_sets, dtype=np.int32)
    num_minimal_sets = 0
    
    for i in range(num_requirement_sets):
        is_minimal = True
        
        # Check if this set is a superset of any other set
        for j in range(num_requirement_sets):
            if i == j:
                continue
                
            # Is set j a subset of set i?
            if requirement_sizes[j] <= requirement_sizes[i]:
                all_j_in_i = True
                for k in range(requirement_sizes[j]):
                    rank_j = requirement_sets[j][k]
                    found_in_i = False
                    for l in range(requirement_sizes[i]):
                        if requirement_sets[i][l] == rank_j:
                            found_in_i = True
                            break
                    if not found_in_i:
                        all_j_in_i = False
                        break
                
                if all_j_in_i and requirement_sizes[j] < requirement_sizes[i]:
                    # Set j is a proper subset of set i, so set i is not minimal
                    is_minimal = False
                    break
        
        if is_minimal:
            for k in range(requirement_sizes[i]):
                minimal_sets[num_minimal_sets][k] = requirement_sets[i][k]
            minimal_sizes[num_minimal_sets] = requirement_sizes[i]
            num_minimal_sets += 1
    
    # Now find all unique ranks from minimal sets
    helpful_ranks = np.zeros(15, dtype=np.uint8)
    
    for i in range(num_minimal_sets):
        for j in range(minimal_sizes[i]):
            rank = minimal_sets[i][j]
            helpful_ranks[rank] = 1
    
    # Count total outs from helpful ranks
    total_outs = 0
    for rank in range(2, 15):
        if helpful_ranks[rank] == 1:
            outs = 4 - rank_counts[rank]
            total_outs += outs
    
    if total_outs == 0:
        return 0.0
    
    # Calculate probability: P(at least one helpful card) = 1 - P(no helpful cards)
    non_outs = remaining_deck_size - total_outs
    
    if non_outs <= 0:
        return 1.0
    
    prob_no_help = 1.0
    for i in range(board_cards_remaining):
        if remaining_deck_size - i <= 0:
            break
        if non_outs - i <= 0:
            prob_no_help = 0.0
            break
        prob_no_help *= float(non_outs - i) / float(remaining_deck_size - i)
    
    return 1.0 - prob_no_help

@njit
def OLD_DELTE_calculate_opponent_straight_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    num_opponents: int
) -> float:
    """
    Calculate probability that at least one opponent will have a straight by river.
    Uses same logic as player function but considers opponent hole card combinations.
    """
    # Count actual board cards
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
    
    board_cards_remaining = 5 - actual_board_cards
    
    if num_opponents <= 0:
        return 0.0
    
    # Count rank occurrences from board only (opponents will add their hole cards)
    board_rank_counts = np.zeros(15, dtype=np.int32)
    my_rank_counts = np.zeros(15, dtype=np.int32)
    
    # Count my cards (these block opponents)
    for r in hole_ranks:
        if r > 1:
            my_rank_counts[r] += 1
    
    # Count board cards
    for i in range(len(board_ranks)):
        if i < len(board_ranks) and board_ranks[i] > 1:
            board_rank_counts[board_ranks[i]] += 1
    
    # Create board presence flags
    board_rank_present = np.zeros(15, dtype=np.uint8)
    for rank in range(2, 15):
        if board_rank_counts[rank] > 0:
            board_rank_present[rank] = 1
    
    # Check if board already has straight (everyone has straight)
    # Check normal straights (2-3-4-5-6 through T-J-Q-K-A)
    for low in range(2, 11):
        consecutive_count = 0
        for i in range(5):
            if board_rank_present[low + i] == 1:
                consecutive_count += 1
        if consecutive_count == 5:
            return 1.0
    
    # Check wheel (A-2-3-4-5)
    wheel_count = 0
    if board_rank_present[14] == 1:  # Ace
        wheel_count += 1
    if board_rank_present[2] == 1:   # 2
        wheel_count += 1
    if board_rank_present[3] == 1:   # 3
        wheel_count += 1
    if board_rank_present[4] == 1:   # 4
        wheel_count += 1
    if board_rank_present[5] == 1:   # 5
        wheel_count += 1
    if wheel_count == 5:
        return 1.0
    
    if board_cards_remaining == 0:
        return 0.0
    
    # Calculate remaining deck size
    cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - cards_seen
    
    # Find opponent straight potential using same logic as player function
    max_opponent_prob = 0.0
    
    # Define all possible straights
    straight_definitions = np.array([
        [14, 2, 3, 4, 5],    # A-2-3-4-5 (wheel)
        [2, 3, 4, 5, 6],     # 2-3-4-5-6
        [3, 4, 5, 6, 7],     # 3-4-5-6-7
        [4, 5, 6, 7, 8],     # 4-5-6-7-8
        [5, 6, 7, 8, 9],     # 5-6-7-8-9
        [6, 7, 8, 9, 10],    # 6-7-8-9-T
        [7, 8, 9, 10, 11],   # 7-8-9-T-J
        [8, 9, 10, 11, 12],  # 8-9-T-J-Q
        [9, 10, 11, 12, 13], # 9-T-J-Q-K
        [10, 11, 12, 13, 14] # T-J-Q-K-A
    ], dtype=np.int32)
    
    # Scenario 1: Opponents need exactly 1 card (immediate straights or 1-card draws)
    for straight_idx in range(10):
        straight_ranks = straight_definitions[straight_idx]
        
        # Count what board provides for this straight
        board_has = 0
        missing_ranks = []
        
        for i in range(5):
            rank = straight_ranks[i]
            if board_rank_present[rank] == 1:
                board_has += 1
            else:
                missing_ranks.append(rank)
        
        need_count = len(missing_ranks)
        
        # Check 1-card scenarios first (highest priority)
        if need_count == 1 and board_cards_remaining >= 1:
            # Opponents need exactly 1 card - could have it in hole cards or get it from board
            missing_rank = missing_ranks[0]
            available = 4 - my_rank_counts[missing_rank] - board_rank_counts[missing_rank]
            
            if available > 0:
                # Probability at least one opponent has this card in hole cards
                prob_one_has = 2.0 * float(available) / float(remaining_deck_size)
                prob_at_least_one = 1.0 - (1.0 - prob_one_has) ** num_opponents
                max_opponent_prob = max(max_opponent_prob, prob_at_least_one)
        
        # Check 2-card scenarios (if no 1-card draws found and enough cards remaining)
        elif need_count == 2 and board_cards_remaining >= 2:
            # Count how many board cards this straight has
            board_cards_count = 0
            for rank in straight_ranks:
                if board_rank_present[rank] == 1:
                    board_cards_count += 1
            
            if board_cards_count >= 3:  # Board provides 3+ cards for this straight
                # Create fixed-size array for board cards in this straight
                board_cards_in_straight = np.zeros(5, dtype=np.int32)
                board_cards_found = 0
                
                for rank in straight_ranks:
                    if board_rank_present[rank] == 1:
                        board_cards_in_straight[board_cards_found] = rank
                        board_cards_found += 1
                
                # Sort the board cards (bubble sort for numba)
                for i in range(board_cards_found):
                    for j in range(i + 1, board_cards_found):
                        if board_cards_in_straight[i] > board_cards_in_straight[j]:
                            temp = board_cards_in_straight[i]
                            board_cards_in_straight[i] = board_cards_in_straight[j]
                            board_cards_in_straight[j] = temp
                
                # Check if board cards are reasonably connected (gaps ≤ 1)
                reasonable_board = True
                for j in range(board_cards_found - 1):
                    gap = board_cards_in_straight[j+1] - board_cards_in_straight[j] - 1
                    if gap > 1:
                        reasonable_board = False
                        break
                
                if reasonable_board:
                    # Calculate probability opponents have both missing ranks in hole cards
                    rank1 = missing_ranks[0]
                    rank2 = missing_ranks[1]
                    avail1 = 4 - my_rank_counts[rank1] - board_rank_counts[rank1]
                    avail2 = 4 - my_rank_counts[rank2] - board_rank_counts[rank2]
                    
                    if avail1 > 0 and avail2 > 0:
                        # Account for future board cards as well
                        future_deck_size = remaining_deck_size - (num_opponents * 2)
                        
                        # Probability one opponent has both in hole cards (immediate)
                        prob_both_hole = float(avail1 * avail2) / float(remaining_deck_size * (remaining_deck_size - 1))
                        prob_at_least_one_both = 1.0 - (1.0 - prob_both_hole) ** num_opponents
                        
                        # Also consider: opponent has 1 card, board provides the other
                        total_prob = prob_at_least_one_both
                        if board_cards_remaining >= 1:
                            prob_one_card = 2.0 * float(avail1 + avail2) / float(remaining_deck_size)
                            prob_someone_has_one = 1.0 - (1.0 - prob_one_card) ** num_opponents
                            
                            # Rough probability board helps (simplified)
                            if future_deck_size > 0:
                                prob_board_helps = float(avail1 + avail2) / float(future_deck_size)
                                combined_prob = prob_someone_has_one * prob_board_helps
                                
                                # Use the better of the two scenarios (avoid double counting)
                                total_prob = prob_at_least_one_both + combined_prob * 0.5
                                if total_prob > 1.0:
                                    total_prob = 1.0
                        
                        if total_prob > max_opponent_prob:
                            max_opponent_prob = total_prob
    
    # Scenario 2: Backdoor potential (only if no strong draws found)
    if max_opponent_prob < 0.1 and board_cards_remaining >= 2:
        for straight_idx in range(10):
            straight_ranks = straight_definitions[straight_idx]
            
            # Count board contribution
            board_has = 0
            for rank in straight_ranks:
                if board_rank_present[rank] == 1:
                    board_has += 1
            
            # If board has exactly 2 cards for this straight, opponents could have backdoor potential
            if board_has == 2:
                # Count missing cards not blocked by me
                available_missing = 0
                for rank in straight_ranks:
                    if board_rank_present[rank] == 0:  # Not on board
                        available = 4 - my_rank_counts[rank] - board_rank_counts[rank]
                        if available > 0:
                            available_missing += available
                
                if available_missing >= 8:  # Reasonable backdoor potential
                    # Very conservative backdoor probability
                    backdoor_prob = float(available_missing) / float(remaining_deck_size * 2)
                    if backdoor_prob > 0.1:
                        backdoor_prob = 0.1
                    backdoor_prob *= float(num_opponents) / 9.0  # Scale with opponent count
                    if backdoor_prob > max_opponent_prob:
                        max_opponent_prob = backdoor_prob
    
    return max_opponent_prob


@njit
def calculate_opponent_straight_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    num_opponents: int
) -> float:
    """
    Calculate probability analytically by enumerating what the Monte Carlo does.
    """
    if num_opponents <= 0:
        return 0.0
    
    # Count actual board cards
    actual_board_cards = 0
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
    
    board_cards_remaining = 5 - actual_board_cards
    
    # Count available cards by rank
    available_by_rank = np.zeros(15, dtype=np.int32)
    for rank in range(2, 15):
        available_by_rank[rank] = 4
    
    # Subtract my hole cards
    for r in hole_ranks:
        if r > 1:
            available_by_rank[r] -= 1
    
    # Subtract board cards
    for i in range(len(board_ranks)):
        if i < len(board_ranks) and board_ranks[i] > 1:
            available_by_rank[board_ranks[i]] -= 1
    
    # Calculate total remaining cards
    total_remaining = 0
    for rank in range(2, 15):
        total_remaining += available_by_rank[rank]
    
    if total_remaining < 2 * num_opponents + board_cards_remaining:
        return 0.0
    
    # Get current board ranks
    current_board = np.zeros(5, dtype=np.int32)
    board_size = 0
    for i in range(len(board_ranks)):
        if i < len(board_ranks) and board_ranks[i] > 1:
            current_board[board_size] = board_ranks[i]
            board_size += 1
    
    # Function to check if ranks contain a straight
    def has_straight_ranks(ranks, num_ranks):
        # Create rank presence array
        rank_present = np.zeros(15, dtype=np.uint8)
        for i in range(num_ranks):
            if ranks[i] > 1:
                rank_present[ranks[i]] = 1
        
        # Check normal straights (2-3-4-5-6 through T-J-Q-K-A)
        for low in range(2, 11):
            consecutive = 0
            for j in range(5):
                if rank_present[low + j] == 1:
                    consecutive += 1
            if consecutive == 5:
                return True
        
        # Check wheel (A-2-3-4-5)
        wheel_count = 0
        if rank_present[14] == 1:  # Ace
            wheel_count += 1
        for rank in range(2, 6):  # 2,3,4,5
            if rank_present[rank] == 1:
                wheel_count += 1
        if wheel_count == 5:
            return True
        
        return False
    
    # Calculate single opponent success probability
    single_opponent_success = 0.0
    
    # Enumerate possible opponent hole card combinations
    for rank1 in range(2, 15):
        for rank2 in range(rank1, 15):  # rank2 >= rank1 to avoid duplicates
            avail1 = available_by_rank[rank1]
            avail2 = available_by_rank[rank2]
            
            if avail1 <= 0 or avail2 <= 0:
                continue
            
            # Calculate probability of this hole card combination
            if rank1 == rank2:
                # Pocket pair
                if avail1 < 2:
                    continue
                prob_hole = (float(avail1) / float(total_remaining)) * \
                           (float(avail1 - 1) / float(total_remaining - 1))
            else:
                # Different ranks  
                prob_hole = 2.0 * (float(avail1) / float(total_remaining)) * \
                           (float(avail2) / float(total_remaining - 1))
            
            # Create test hand with these hole cards
            test_hand = np.zeros(7, dtype=np.int32)
            test_hand[0] = rank1
            test_hand[1] = rank2
            
            # Add current board
            for i in range(board_size):
                test_hand[2 + i] = current_board[i]
            
            # Calculate probability this leads to a straight
            if board_cards_remaining == 0:
                # River case - just check current hand
                if has_straight_ranks(test_hand, 2 + board_size):
                    single_opponent_success += prob_hole
            
            elif board_cards_remaining == 1:
                # Turn case - check all possible river cards
                remaining_after_hole = total_remaining - 2
                straight_outcomes = 0
                total_outcomes = 0
                
                for river_rank in range(2, 15):
                    river_avail = available_by_rank[river_rank]
                    
                    # Adjust for hole cards used
                    if river_rank == rank1:
                        if rank1 == rank2:
                            river_avail -= 2
                        else:
                            river_avail -= 1
                    elif river_rank == rank2:
                        river_avail -= 1
                    
                    if river_avail > 0:
                        total_outcomes += river_avail
                        
                        # Test this river card
                        test_hand[2 + board_size] = river_rank
                        if has_straight_ranks(test_hand, 2 + board_size + 1):
                            straight_outcomes += river_avail
                
                if total_outcomes > 0:
                    prob_river_helps = float(straight_outcomes) / float(total_outcomes)
                    single_opponent_success += prob_hole * prob_river_helps
            
            elif board_cards_remaining == 2:
                # Flop case - sample turn/river combinations
                remaining_after_hole = total_remaining - 2
                
                if remaining_after_hole < 2:
                    continue
                
                helpful_combinations = 0.0
                total_combinations = 0.0
                
                # Sample key turn/river combinations
                for turn_rank in range(2, 15):
                    turn_avail = available_by_rank[turn_rank]
                    
                    # Adjust for hole cards
                    if turn_rank == rank1:
                        if rank1 == rank2:
                            turn_avail -= 2
                        else:
                            turn_avail -= 1
                    elif turn_rank == rank2:
                        turn_avail -= 1
                    
                    if turn_avail <= 0:
                        continue
                    
                    for river_rank in range(2, 15):
                        river_avail = available_by_rank[river_rank]
                        
                        # Adjust for hole cards and turn card
                        if river_rank == rank1:
                            if rank1 == rank2:
                                river_avail -= 2
                            else:
                                river_avail -= 1
                        elif river_rank == rank2:
                            river_avail -= 1
                        
                        if river_rank == turn_rank:
                            river_avail -= 1
                        
                        if river_avail <= 0:
                            continue
                        
                        # Weight by probability of this combination
                        combo_weight = float(turn_avail * river_avail)
                        total_combinations += combo_weight
                        
                        # Test if this makes a straight
                        test_hand[2 + board_size] = turn_rank
                        test_hand[2 + board_size + 1] = river_rank
                        if has_straight_ranks(test_hand, 2 + board_size + 2):
                            helpful_combinations += combo_weight
                
                if total_combinations > 0.0:
                    prob_board_helps = helpful_combinations / total_combinations
                    single_opponent_success += prob_hole * prob_board_helps
    
    # Cap single opponent probability
    if single_opponent_success > 1.0:
        single_opponent_success = 1.0
    
    # Calculate final probability for multiple opponents
    prob_all_fail = (1.0 - single_opponent_success) ** num_opponents
    final_prob = 1.0 - prob_all_fail
    
    return final_prob








def encode_game_state(
    hole_cards: list[tuple[int, int]],
    board_cards: list[tuple[int, int]],
    num_players: int
) -> np.ndarray:
    features = []

    hole_suits = np.empty(2, dtype=np.int32)
    hole_ranks = np.empty(2, dtype=np.int32)

    # --- Hole card encoding ---
    for i, (rank, suit) in enumerate(hole_cards):
        card_onehot = [0] * 52
        index = (rank - 2) * 4 + suit
        card_onehot[index] = 1
        features.extend(card_onehot)
        hole_suits[i] = suit
        hole_ranks[i] = rank

    # --- Board card encoding (pad to 5 cards) ---
    board_suits = np.full(5, -1, dtype=np.int32)
    board_ranks = np.zeros(5, dtype=np.int32)
    for i in range(5):
        if i < len(board_cards):
            rank, suit = board_cards[i]
            card_onehot = [0] * 52
            index = (rank - 2) * 4 + suit
            card_onehot[index] = 1
            board_suits[i] = suit
            board_ranks[i] = rank
        else:
            card_onehot = [0] * 52
        features.extend(card_onehot)

    # --- Player count continuous
    player_count_normalized = num_players / 10.0
    features.append(player_count_normalized)

    # --- Street indicator FIRST (more logical ordering) ---
    street = len(board_cards) / 5.0
    features.append(street)

    # --- Feature extraction with fixed arrays ---
    flush_feats = extract_flush_features(hole_suits, board_suits)
    features.extend(flush_feats.tolist())

    rank_feats = extract_rank_features(hole_ranks, board_ranks)
    features.extend(rank_feats.tolist())
    
    straight_feats = extract_straight_features(hole_ranks, board_ranks)
    features.extend(straight_feats.tolist())
    
    context_feats = extract_contextual_features(hole_ranks, board_ranks)
    features.extend(context_feats.tolist())
    
    board_feats = extract_board_features(hole_ranks, board_ranks)
    features.extend(board_feats.tolist())
    
    # NEW: Interaction features using results from above
    interaction_feats = extract_interaction_features(
        hole_suits, hole_ranks, board_suits, board_ranks,
        flush_feats[5],      # flush_possible_with_hole
        flush_feats[6],      # backdoor_flush_draw
        straight_feats[3],   # straight_possible_with_hole  
        straight_feats[4],   # open_ended
        straight_feats[5]    # gutshot
    )
    features.extend(interaction_feats.tolist())
    
    # NEW: Single hand strength scalar (replaces the 5-feature version)
    hand_strength = extract_hand_strength_scalar(hole_ranks, board_ranks, hole_suits, board_suits)
    features.append(hand_strength)

    board_texture = extract_simple_board_texture(board_ranks)
    features.append(board_texture)
    
    dominated_feats = extract_dominated_draws(
        hole_ranks, hole_suits,
        flush_feats[5],  # flush_possible_with_hole
        straight_feats[3]  # straight_possible_with_hole
    )
    features.extend(dominated_feats.tolist())
    
    board_vol_feats = extract_board_volume_features(board_ranks, board_suits)
    features.extend(board_vol_feats.tolist())
    
    blocker_feats = extract_flush_blocker_count(hole_suits, board_suits)
    features.extend(blocker_feats.tolist())
    
    advanced_feats = extract_advanced_poker_features(
        hole_ranks, hole_suits, board_ranks, board_suits
    )
    features.extend(advanced_feats.tolist())
    
    # --- Relative Board Contribution (NEW FEATURE) ---
    board_contrib_feat = extract_relative_board_contribution(
        hole_ranks, hole_suits, board_ranks, board_suits
    )
    features.extend(board_contrib_feat.tolist())
    
    # --- Player zero pair probabilities for each rank (2 to 14), opponents, and deltas ---
    for card_rank in range(2, 15):
        player_zero_prob = calculate_pair_probability(
            hole_ranks=hole_ranks,
            hole_suits=hole_suits,
            board_ranks=board_ranks,
            board_suits=board_suits,
            card_rank=card_rank
        )
        features.append(player_zero_prob)
        
        opp_prob = calculate_opponent_pair_probability(
            hole_ranks=hole_ranks,
            hole_suits=hole_suits,
            board_ranks=board_ranks,
            board_suits=board_suits,
            card_rank=card_rank,
            num_opponents=num_players-1
        )
        features.append(opp_prob)
        
        # Normalize delta from [-1,1] to [0,1]
        delta = player_zero_prob - opp_prob
        normalized_delta = (delta + 1.0) / 2.0
        features.append(normalized_delta)

        
        
    # Add these loops after your pair probabilities
    for card_rank in range(2, 15):
        # Self trips/quads
        trips_prob = calculate_trips_probability(hole_ranks, hole_suits, board_ranks, board_suits, card_rank)
        quads_prob = calculate_quads_probability(hole_ranks, hole_suits, board_ranks, board_suits, card_rank)
        features.extend([trips_prob, quads_prob])
        
        # Opponent trips/quads  
        opp_trips = calculate_opponent_trips_probability(hole_ranks, hole_suits, board_ranks, board_suits, card_rank, num_players - 1)
        opp_quads = calculate_opponent_quads_probability(hole_ranks, hole_suits, board_ranks, board_suits, card_rank, num_players - 1)
        features.extend([opp_trips, opp_quads])
        
        # Delta features (normalized to [0,1])
        trips_delta = (trips_prob - opp_trips + 1.0) / 2.0
        quads_delta = (quads_prob - opp_quads + 1.0) / 2.0
        features.extend([trips_delta, quads_delta])
        
        
    # --- Full house probabilities ---
    fullhouse_prob = calculate_fullhouse_probability(
        hole_ranks, hole_suits, board_ranks, board_suits
    )
    opp_fullhouse_prob = calculate_opponent_fullhouse_probability(
        hole_ranks, hole_suits, board_ranks, board_suits, num_players - 1
    )
    fullhouse_delta = (fullhouse_prob - opp_fullhouse_prob + 1.0) / 2.0  # normalized to [0, 1]
    features.extend([fullhouse_prob, opp_fullhouse_prob, fullhouse_delta])

    # --- Flush probabilities ---
    my_flush_prob = calculate_flush_probability(
        hole_ranks, hole_suits, board_ranks, board_suits
    )
    opp_flush_prob = calculate_opponent_flush_probability(
        hole_ranks, hole_suits, board_ranks, board_suits, num_players - 1
    )
    flush_delta = (my_flush_prob - opp_flush_prob + 1.0) / 2.0  # normalized to [0, 1]
    
    features.extend([my_flush_prob, opp_flush_prob, flush_delta])
            
    # --- Dense card representation (7 features) ---
    dense_card_features = create_dense_card_representation(hole_ranks, hole_suits, board_ranks, board_suits)
    features.extend(dense_card_features.tolist())

    return np.array(features, dtype=np.float32)





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




"""


hole_cards = ['4c', '5c']
board_cards = []

hole_tuples = [parse_card(c) for c in hole_cards]
board_tuples = [parse_card(c) for c in board_cards]

hole_ranks = np.array([h[0] for h in hole_tuples], dtype=np.int32)
hole_suits = np.array([h[1] for h in hole_tuples], dtype=np.int32)
board_ranks = np.zeros(5, dtype=np.int32)
board_suits = np.full(5, -1, dtype=np.int32)

for i, (rank, suit) in enumerate(board_tuples):
    board_ranks[i] = rank
    board_suits[i] = suit

my_prob = calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)
opp_prob = calculate_opponent_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits, 5)

print(f"Test 1 - Suited hole cards preflop:")
print(f"My flush probability: {my_prob:.4f}")
print(f"Opponent flush probability: {opp_prob:.4f}")
print()

# Test Case 2: Flush draw on flop (4 clubs)
hole_cards = ['4c', '5c']
board_cards = ['Ac', '7c', 'Ts']

hole_tuples = [parse_card(c) for c in hole_cards]
board_tuples = [parse_card(c) for c in board_cards]

hole_ranks = np.array([h[0] for h in hole_tuples], dtype=np.int32)
hole_suits = np.array([h[1] for h in hole_tuples], dtype=np.int32)
board_ranks = np.zeros(5, dtype=np.int32)
board_suits = np.full(5, -1, dtype=np.int32)

for i, (rank, suit) in enumerate(board_tuples):
    board_ranks[i] = rank
    board_suits[i] = suit

my_prob = calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)
opp_prob = calculate_opponent_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits, 5)

print(f"Test 2 - 4-card flush draw on flop:")
print(f"My flush probability: {my_prob:.4f}")  # Should be high (~35%)
print(f"Opponent flush probability: {opp_prob:.4f}")
print()




# Test Case 3: Already made flush
hole_cards = ['4c', '5c']
board_cards = ['Ac', '7c', '9c']

hole_tuples = [parse_card(c) for c in hole_cards]
board_tuples = [parse_card(c) for c in board_cards]

hole_ranks = np.array([h[0] for h in hole_tuples], dtype=np.int32)
hole_suits = np.array([h[1] for h in hole_tuples], dtype=np.int32)
board_ranks = np.zeros(5, dtype=np.int32)
board_suits = np.full(5, -1, dtype=np.int32)

for i, (rank, suit) in enumerate(board_tuples):
    board_ranks[i] = rank
    board_suits[i] = suit

my_prob = calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)
opp_prob = calculate_opponent_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits, 5)

print(f"Test 3 - Already made flush:")
print(f"My flush probability: {my_prob:.4f}")  # Should be 1.0
print(f"Opponent flush probability: {opp_prob:.4f}")
print()

# Test Case 4: No flush potential
hole_cards = ['4c', '5h']
board_cards = ['As', '7d', 'Ts']

hole_tuples = [parse_card(c) for c in hole_cards]
board_tuples = [parse_card(c) for c in board_cards]

hole_ranks = np.array([h[0] for h in hole_tuples], dtype=np.int32)
hole_suits = np.array([h[1] for h in hole_tuples], dtype=np.int32)
board_ranks = np.zeros(5, dtype=np.int32)
board_suits = np.full(5, -1, dtype=np.int32)

for i, (rank, suit) in enumerate(board_tuples):
    board_ranks[i] = rank
    board_suits[i] = suit

my_prob = calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)
opp_prob = calculate_opponent_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits, 5)

print(f"Test 4 - Rainbow board, no flush potential:")
print(f"My flush probability: {my_prob:.4f}")  # Should be 0.0
print(f"Opponent flush probability: {opp_prob:.4f}")
print()

# Test Case 5: Board flush possible for opponents
hole_cards = ['4c', '5h']  # No clubs
board_cards = ['Ac', '7c', '9c']  # 3 clubs on board

hole_tuples = [parse_card(c) for c in hole_cards]
board_tuples = [parse_card(c) for c in board_cards]

hole_ranks = np.array([h[0] for h in hole_tuples], dtype=np.int32)
hole_suits = np.array([h[1] for h in hole_tuples], dtype=np.int32)
board_ranks = np.zeros(5, dtype=np.int32)
board_suits = np.full(5, -1, dtype=np.int32)

for i, (rank, suit) in enumerate(board_tuples):
    board_ranks[i] = rank
    board_suits[i] = suit

my_prob = calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)
opp_prob = calculate_opponent_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits, 5)

print(f"Test 5 - 3 clubs on board, I have no clubs:")
print(f"My flush probability: {my_prob:.4f}")  # Should be 0.0
print(f"Opponent flush probability: {opp_prob:.4f}")  # Should be moderate


# Test Case 6: Flush draw on 1 pocket card
hole_cards = ['4c', '5d']
board_cards = ['Ac', '7c', 'Qc']

hole_tuples = [parse_card(c) for c in hole_cards]
board_tuples = [parse_card(c) for c in board_cards]

hole_ranks = np.array([h[0] for h in hole_tuples], dtype=np.int32)
hole_suits = np.array([h[1] for h in hole_tuples], dtype=np.int32)
board_ranks = np.zeros(5, dtype=np.int32)
board_suits = np.full(5, -1, dtype=np.int32)

for i, (rank, suit) in enumerate(board_tuples):
    board_ranks[i] = rank
    board_suits[i] = suit

my_prob = calculate_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits)
opp_prob = calculate_opponent_flush_probability(hole_ranks, hole_suits, board_ranks, board_suits, 5)

print(f"Test 2 - 4-card flush draw on flop:")
print(f"My flush probability: {my_prob:.4f}")  # Should be high (~35%)
print(f"Opponent flush probability: {opp_prob:.4f}")
print()
"""



def debug_opponent_flush_probability(
    hole_ranks: np.ndarray,
    hole_suits: np.ndarray, 
    board_ranks: np.ndarray,
    board_suits: np.ndarray,
    num_opponents: int
) -> float:
    """Debug version to see what's happening"""
    
    # Count actual board cards
    actual_board_cards = 0
    board_suit_counts = np.zeros(4, dtype=np.int32)
    
    for i in range(len(board_ranks)):
        if board_ranks[i] > 1:
            actual_board_cards += 1
            if board_suits[i] >= 0:
                board_suit_counts[board_suits[i]] += 1
    
    print(f"Board suit counts: {board_suit_counts}")  # Should be [0, 0, 1, 2] for [clubs, hearts, diamonds, spades]
    
    # Count my cards by suit  
    my_suit_counts = np.zeros(4, dtype=np.int32)
    for i in range(len(hole_suits)):
        if hole_suits[i] >= 0 and hole_ranks[i] > 1:
            my_suit_counts[hole_suits[i]] += 1
    
    print(f"My suit counts: {my_suit_counts}")  # Should be [1, 1, 0, 0] for [clubs, hearts, diamonds, spades]
    
    cards_seen = len(hole_ranks) + actual_board_cards
    remaining_deck_size = 52 - cards_seen
    board_cards_remaining = 5 - actual_board_cards
    
    print(f"Remaining deck: {remaining_deck_size}, Board cards remaining: {board_cards_remaining}")
    
    max_opponent_flush_prob = 0.0
    
    for suit in range(4):
        total_suit_cards_seen = my_suit_counts[suit] + board_suit_counts[suit]
        suit_cards_remaining = 13 - total_suit_cards_seen
        
        print(f"\nSuit {suit}: seen={total_suit_cards_seen}, remaining={suit_cards_remaining}")
        
        if suit_cards_remaining < 2:
            print(f"  Skipping suit {suit} - not enough cards")
            continue
        
        # Main scenario: Suited hole cards + board completes
        prob_one_suited = (
            float(suit_cards_remaining) * float(suit_cards_remaining - 1) /
            (float(remaining_deck_size) * float(remaining_deck_size - 1))
        )
        prob_at_least_one_suited = 1.0 - (1.0 - prob_one_suited) ** num_opponents
        
        print(f"  Prob one opponent suited: {prob_one_suited:.6f}")
        print(f"  Prob at least one suited: {prob_at_least_one_suited:.6f}")
        
        # Given suited hole cards, probability board completes flush
        total_with_suited = board_suit_counts[suit] + 2
        need_from_board = max(0, 5 - total_with_suited)
        
        print(f"  Total with suited: {total_with_suited}, need from board: {need_from_board}")
        
        if need_from_board <= 0:
            prob_board_completes = 1.0
            print(f"  Already have flush!")
        elif need_from_board <= board_cards_remaining:
            future_deck_size = remaining_deck_size - (num_opponents * 2)
            future_suit_cards = suit_cards_remaining - 2
            
            print(f"  Future deck: {future_deck_size}, future suit cards: {future_suit_cards}")
            
            if need_from_board == 2 and future_suit_cards >= 2 and future_deck_size >= 2:
                prob_board_completes = (
                    float(future_suit_cards) * float(future_suit_cards - 1) /
                    (float(future_deck_size) * float(future_deck_size - 1))
                )
                print(f"  Board completes prob (need 2): {prob_board_completes:.6f}")
            else:
                prob_board_completes = 0.0
                print(f"  Board completes prob: 0 (impossible)")
        else:
            prob_board_completes = 0.0
            print(f"  Board completes prob: 0 (not enough cards)")
        
        suit_flush_prob = prob_at_least_one_suited * prob_board_completes
        print(f"  Suit {suit} total prob: {suit_flush_prob:.6f}")
        
        max_opponent_flush_prob = max(max_opponent_flush_prob, suit_flush_prob)
    
    print(f"\nFinal max prob: {max_opponent_flush_prob:.6f}")
    return max_opponent_flush_prob

