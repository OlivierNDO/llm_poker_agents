"""
src/monte_carlo_tracker.py

Track Monte Carlo usage per player per hand
"""

class MonteCarloUsageTracker:
    """Track Monte Carlo usage with 1-use-per-hand limit"""
    
    def __init__(self, players):
        self.usage = {player.name: 0 for player in players}
        self.max_uses = 1
    
    def can_use_monte_carlo(self, player_name: str) -> bool:
        """Check if player can use Monte Carlo"""
        return self.usage.get(player_name, 0) < self.max_uses
    
    def record_usage(self, player_name: str):
        """Record that player used Monte Carlo"""
        if player_name not in self.usage:
            self.usage[player_name] = 0
        self.usage[player_name] += 1
    
    def get_usage_status(self, player_name: str) -> str:
        """Get usage status string for player"""
        used = self.usage.get(player_name, 0)
        remaining = max(0, self.max_uses - used)
        return f"Monte Carlo: {remaining}/{self.max_uses} uses remaining"
    
    def reset_for_new_hand(self, players):
        """Reset usage for new hand"""
        self.usage = {player.name: 0 for player in players}