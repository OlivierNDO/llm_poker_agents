"""
feature_engineering.py

x = encode_game_state(
    hole_cards=[(14, 2), (13, 2)],           # A♦ K♦
    board_cards=[(2, 2), (7, 2), (9, 2)],     # 2♦ 7♦ 9♦
    num_players=6
)
print(x.shape)

"""
from typing import List

from numba import njit
import numpy as np

@njit
def extract_flush_features(hole_suits: np.ndarray, board_suits: np.ndarray) -> np.ndarray:
    suited_hole = 1.0 if hole_suits[0] == hole_suits[1] else 0.0

    suit_counts = np.zeros(4, dtype=np.int32)
    for s in board_suits:
        if s >= 0:  # Skip zero padding
            suit_counts[s] += 1
            
    board_2flush = 0.0  # NEW: Backdoor potential
    board_3flush = 0.0
    board_4flush = 0.0
    board_5flush = 0.0
    flush_possible_with_hole = 0.0
    backdoor_flush_draw = 0.0  # NEW: Backdoor draw
    flush_blocker = 0.0

    for suit in range(4):
        count = suit_counts[suit]
        num_in_hand = int(hole_suits[0] == suit) + int(hole_suits[1] == suit)
        
        if count == 2:
            board_2flush = 1.0
            # Backdoor flush draw = 2 on board + 1 in hand, need 2 more
            if num_in_hand >= 1:
                backdoor_flush_draw = 1.0
        elif count >= 3:
            if count == 3:
                board_3flush = 1.0
            elif count == 4:
                board_4flush = 1.0
            elif count >= 5:
                board_5flush = 1.0

            if count + num_in_hand >= 5:
                flush_possible_with_hole = 1.0
            if num_in_hand > 0:
                flush_blocker = 1.0

    return np.array([
        suited_hole,
        board_2flush,           # NEW
        board_3flush,
        board_4flush,
        board_5flush,
        flush_possible_with_hole,
        backdoor_flush_draw,    # NEW
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
def extract_straight_features(hole_ranks: np.ndarray, board_ranks: np.ndarray) -> np.ndarray:
    """
    Efficiently extract straight-related features.
    """
    # Create rank_flags for all cards (hole + board)
    rank_flags = np.zeros(15, dtype=np.uint8)
    for r in hole_ranks:
        rank_flags[r] = 1
        if r == 14:
            rank_flags[1] = 1  # Ace as low
    
    for i in range(5):
        r = board_ranks[i]
        if r > 1:  # Skip zero padding
            rank_flags[r] = 1
            if r == 14:
                rank_flags[1] = 1

    # Create board_flags for board-only features
    board_flags = np.zeros(15, dtype=np.uint8)
    for i in range(5):
        r = board_ranks[i]
        if r > 1:  # Skip zero padding
            board_flags[r] = 1
            if r == 14:
                board_flags[1] = 1

    # Board straight detection
    board_3straight = 0
    board_4straight = 0
    board_5straight = 0
    for low in range(1, 11):
        span = board_flags[low:low+5]
        total = np.sum(span)
        if total >= 3:
            board_3straight = 1
        if total >= 4:
            board_4straight = 1
        if total == 5:
            board_5straight = 1
            break

    # Straight possible with hole cards
    straight_possible_with_hole = 0
    for low in range(1, 11):
        if np.sum(rank_flags[low:low+5]) == 5:
            straight_possible_with_hole = 1
            break

    # Open-ended & gutshot logic
    open_ended = 0
    gutshot = 0
    for low in range(1, 11):
        window = rank_flags[low:low+5]
        total = np.sum(window)
        if total == 4:
            missing = 0
            for i in range(5):
                if window[i] == 0:
                    missing = i
                    break
            if missing == 0 or missing == 4:
                open_ended = 1
            else:
                gutshot = 1

    return np.array([
        float(board_3straight),
        float(board_4straight),
        float(board_5straight),
        float(straight_possible_with_hole),
        float(open_ended),
        float(gutshot)
    ], dtype=np.float32)


@njit
def extract_contextual_features(hole_ranks: np.ndarray, board_ranks: np.ndarray) -> np.ndarray:
    # Flag if hole cards form a pair
    hole_pair = 1.0 if hole_ranks[0] == hole_ranks[1] else 0.0

    # Max rank on the board
    max_board_rank = 0
    for r in board_ranks:
        if r > max_board_rank:  # This naturally skips zeros
            max_board_rank = r

    # Count how many hole cards are higher than all board cards
    overcards = 1.0 if (hole_ranks[0] > max_board_rank and hole_ranks[1] > max_board_rank) else 0.0

    # Best hole card's rank
    kicker_rank = max(hole_ranks[0], hole_ranks[1]) / 14.0

    # Top board card (scaled)
    top_board = max_board_rank / 14.0

    # Count hole cards above board
    above_board = 0
    for r in hole_ranks:
        if r > max_board_rank:
            above_board += 1

    # Determine if hole cards improve hand over board-only
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
                return 7  # Quads
            elif counts[rank] == 3:
                for r2 in range(14, 1, -1):
                    if r2 != rank and counts[r2] >= 2:
                        return 6  # Full house
                return 3  # Trips
            elif counts[rank] == 2:
                for r2 in range(rank - 1, 1, -1):
                    if counts[r2] == 2:
                        return 2  # Two pair
                return 1  # One pair
        return 0  # High card

    board_strength = get_best_rank(rank_counts_board)
    full_strength = get_best_rank(rank_counts_full)

    hole_improves_board = 1.0 if full_strength > board_strength else 0.0
    
    # Add gap between hole cards
    gap = abs(hole_ranks[0] - hole_ranks[1]) - 1
    gap = max(0, gap)  # Ensure non-negative
    gap_normalized = min(gap, 12) / 12.0  # Cap at 12 (A-2), normalize

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
    board_rank_counts = np.zeros(15, dtype=np.int32)
    for r in board_ranks:
        if r > 1:  # Skip zero padding
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
    
    # Count different rank types on board
    has_pair = 0
    has_trips = 0
    
    for rank in range(2, 15):
        count = board_rank_counts[rank]
        if count == 2:
            board_paired = 1
            has_pair = 1
        elif count == 3:
            board_trips = 1
            has_trips = 1
        elif count == 4:
            board_quads = 1
            has_trips = 1  # Quads contain trips for full house purposes
    
    # Board has full house if it has both trips and a separate pair
    if has_trips == 1 and has_pair == 1:
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
    
    # Check if hole cards + board form a full house
    combined = board_rank_counts + hole_rank_counts
    pair_count = 0
    trips_count = 0
    for count in combined:
        if count >= 2:
            pair_count += 1
        if count >= 3:
            trips_count += 1
    
    # Need at least one trips AND at least one pair (could be same rank if quads+)
    if trips_count >= 1 and pair_count >= 1:
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


@njit
def extract_hand_strength_scalar(hole_ranks: np.ndarray, board_ranks: np.ndarray, 
                                hole_suits: np.ndarray, board_suits: np.ndarray) -> float:
    """
    Single continuous hand strength measure (0-1 scale)
    FINAL FIXED VERSION - STRAIGHT FLUSH VERIFIED
    """
    all_ranks = np.zeros(15, dtype=np.int32)
    for r in hole_ranks:
        all_ranks[r] += 1
    for r in board_ranks:
        if r > 1:  # Skip zero padding
            all_ranks[r] += 1
    
    # OLD AND BROKEN
    suit_counts = np.zeros(4, dtype=np.int32)
    for i in range(5):  # Check each board position
        if i < len(board_suits) and board_suits[i] >= 0 and board_ranks[i] > 1:  # Valid card only
            suit_counts[board_suits[i]] += 1
    for s in hole_suits:
        if s >= 0:  # Valid suit
            suit_counts[s] += 1
    
    # FIXED: Check for ACTUAL straight flush (same 5 cards form both straight and flush)
    has_straight_flush = 0
    straight_flush_high = 0
    
    for suit in range(4):
        if suit_counts[suit] >= 5:
            # Create rank flags for this specific suit only
            suited_rank_flags = np.zeros(15, dtype=np.uint8)
            
            # Add hole cards of this suit
            if hole_suits[0] == suit:
                suited_rank_flags[hole_ranks[0]] = 1
                if hole_ranks[0] == 14:
                    suited_rank_flags[1] = 1  # Ace low
            if hole_suits[1] == suit:
                suited_rank_flags[hole_ranks[1]] = 1
                if hole_ranks[1] == 14:
                    suited_rank_flags[1] = 1  # Ace low
            
            # Add board cards of this suit
            for i in range(5):
                if i < len(board_suits) and board_suits[i] == suit and board_ranks[i] > 1:
                    suited_rank_flags[board_ranks[i]] = 1
                    if board_ranks[i] == 14:
                        suited_rank_flags[1] = 1  # Ace low
            
            # Check for straight within this suit
            for low in range(1, 11):
                if np.sum(suited_rank_flags[low:low+5]) == 5:
                    has_straight_flush = 1
                    straight_flush_high = low + 4 if low > 1 else 5
                    break
            
            if has_straight_flush:
                break
    
    # Regular flush detection (unchanged)
    has_flush = 0
    flush_high = 0
    for suit in range(4):
        if suit_counts[suit] >= 5:
            has_flush = 1
            # Find highest card in this flush suit
            for i in range(5):  # Check all board positions
                if i < len(board_suits) and board_suits[i] == suit and board_ranks[i] > 1:
                    flush_high = max(flush_high, board_ranks[i])
            if hole_suits[0] == suit:
                flush_high = max(flush_high, hole_ranks[0])
            if hole_suits[1] == suit:
                flush_high = max(flush_high, hole_ranks[1])
            break
    
    # Regular straight detection (unchanged)
    rank_flags = np.zeros(15, dtype=np.uint8)
    for r in hole_ranks:
        rank_flags[r] = 1
        if r == 14:
            rank_flags[1] = 1  # Ace low
    for r in board_ranks:
        if r > 1:  # Skip zero padding
            rank_flags[r] = 1
            if r == 14:
                rank_flags[1] = 1  # Ace low
    
    has_straight = 0
    straight_high = 0
    for low in range(1, 11):
        if np.sum(rank_flags[low:low+5]) == 5:
            has_straight = 1
            straight_high = low + 4 if low > 1 else 5  # Handle ace-low straight
            break
    
    # Find hand type ranks (unchanged)
    quads = 0
    trips = 0
    pairs_found = []
    
    for rank in range(2, 15):  # Check all ranks
        count = all_ranks[rank]
        if count == 4:
            quads = rank
        elif count == 3:
            trips = rank
        elif count == 2:
            pairs_found.append(rank)
    
    # Sort pairs by rank (highest first)
    pairs_found.sort(reverse=True)
    pair1 = pairs_found[0] if len(pairs_found) >= 1 else 0
    pair2 = pairs_found[1] if len(pairs_found) >= 2 else 0
    
    # Calculate strength - FIXED STRAIGHT FLUSH
    strength = 0.0
    
    # FIXED: Only true straight flush counts as 800+
    if has_straight_flush:
        strength = 800.0 + straight_flush_high * 1.0  # 800-814
    
    # Rest of the logic unchanged...
    # Quads: 700-799 range  
    elif quads > 0:
        kicker = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                kicker = r
                break
        strength = 700.0 + quads * 6.0 + kicker * 0.1
    
    # Full house - ensure no overlap  
    elif trips > 0 and pair1 > 0:
        strength = 600.0 + trips * 5.0 + pair1 * 0.1
    
    # Flush - improved to consider second card  
    elif has_flush:
        flush_cards = []
        flush_suit = 0
        
        # Find the flush suit
        for suit in range(4):
            if suit_counts[suit] >= 5:
                flush_suit = suit
                break
        
        # Collect flush cards
        for i in range(5):
            if i < len(board_suits) and board_suits[i] == flush_suit and board_ranks[i] > 1:
                flush_cards.append(board_ranks[i])
        if hole_suits[0] == flush_suit:
            flush_cards.append(hole_ranks[0])
        if hole_suits[1] == flush_suit:
            flush_cards.append(hole_ranks[1])
        
        flush_cards.sort(reverse=True)
        flush_cards = flush_cards[:5]
        second_flush = flush_cards[1] if len(flush_cards) > 1 else 0
        third_flush = flush_cards[2] if len(flush_cards) > 2 else 0
        
        strength = 500.0 + flush_cards[0] * 6.0 + second_flush * 0.6 + third_flush * 0.06
    
    # Straight: 400-499 range
    elif has_straight:
        strength = 400.0 + straight_high * 6.0
    
    # Trips: 300-399 range
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
    
    # Two pair: 200-299 range
    elif len(pairs_found) >= 2:
        kicker = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                kicker = r
                break
        strength = 200.0 + pair1 * 6.0 + pair2 * 0.5 + kicker * 0.01
    
    # One pair: 100-199 range
    elif pair1 > 0:
        kicker = 0
        for r in range(14, 1, -1):
            if all_ranks[r] == 1:
                kicker = r
                break
        strength = 100.0 + pair1 * 6.0 + kicker * 0.1
    
    # High card: 0-99 range
    else:
        high = max(hole_ranks[0], hole_ranks[1])
        second = min(hole_ranks[0], hole_ranks[1])
        strength = high * 6.0 + second * 0.1
    
    # Normalize to 0-1
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
    Returns a 1D array with a single value:
    [relative_board_contribution] = board_strength / hand_strength

    Interprets how much of our hand strength comes from the board.
    """
    strength_full = extract_hand_strength_scalar(hole_ranks, board_ranks, hole_suits, board_suits)

    # Fake hole cards = invalid/blank cards (won't contribute to strength)
    blank_hole_ranks = np.array([-1, -1], dtype=np.int32)
    blank_hole_suits = np.array([-1, -1], dtype=np.int32)
    
    strength_board = extract_hand_strength_scalar(blank_hole_ranks, board_ranks, blank_hole_suits, board_suits)

    # Avoid division by zero using fused min/max (safe in Numba)
    denom = strength_full if strength_full > 1e-6 else 1e-6
    relative = strength_board / denom
    return np.array([min(1.0, relative)], dtype=np.float32)


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

    return np.array(features, dtype=np.float32)

