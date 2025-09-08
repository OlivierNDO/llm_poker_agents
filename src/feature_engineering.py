"""
feature_engineering.py
"""
from typing import List, Sequence
from collections import Counter
from numba import njit
import numpy as np




def best_board_pattern(board_ranks: Sequence[int], board_suits: Sequence[str]) -> str:
    """
    Return the single strongest made-hand pattern visible on the board itself.
    Always reports only the best category (no overlapping labels).
    Works for 3–5 community cards.
    """
    n = len(board_ranks)
    rank_counts = Counter(board_ranks)
    suit_counts = Counter(board_suits)

    # --- check flush potential ---
    flush_suit = None
    if any(count >= 5 for count in suit_counts.values()):
        flush_suit = max(suit_counts, key=suit_counts.get)

    # --- straight detection helper ---
    def has_straight(ranks):
        uniq = sorted(set(ranks))
        if len(uniq) < 5:
            return False
        # wheel
        if set([14, 2, 3, 4, 5]).issubset(uniq):
            return True
        for i in range(len(uniq) - 4):
            if uniq[i+4] - uniq[i] == 4:
                return True
        return False

    # --- check straight/straight flush ---
    straight_on_board = has_straight(board_ranks)
    straight_flush_on_board = False
    if flush_suit and n >= 5:
        suited = [r for r, s in zip(board_ranks, board_suits) if s == flush_suit]
        if has_straight(suited):
            straight_flush_on_board = True

    # --- now check in proper ranking order ---
    if straight_flush_on_board:
        return "straight flush"
    if 4 in rank_counts.values():
        return "quads"
    if 3 in rank_counts.values() and 2 in rank_counts.values():
        return "full house"
    if flush_suit:
        return "flush"
    if straight_on_board:
        return "straight"
    if 3 in rank_counts.values():
        return "trips"
    if list(rank_counts.values()).count(2) >= 1:
        return "pair"
    return None









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



