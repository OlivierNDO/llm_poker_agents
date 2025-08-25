# Poker LLM Agent – Project Plan

This document outlines the planned design for the Poker LLM Agent.  
The core idea is that **before deciding on a final action (fold / check / call / bet / raise / all-in)**,  
the agent has the *option* to perform intermediate reasoning or call specialized services/tools.  
These are optional steps – the agent may skip them if it deems them unnecessary.

---

## Preliminary Optional Steps

### 1. Bluff Assessment
- **When triggered**: At the start of the agent’s turn, if one or more opponents have bet this round.
- **Goal**: Estimate the likelihood that each betting opponent is bluffing.
- **Inputs**:
  - Opponents’ recent actions (`GameState.get_actions_since_board_change`)
  - Opponent profiles (`StatsTracker.get_opponent_stats_summary`)
- **Output**: Probability or qualitative estimate (`likely bluffing`, `unlikely bluffing`).
- **Usage**: Provides a signal to inform call/fold/raise decisions.

---

### 2. Bluff Decision
- **When triggered**: Only if the agent has a weak/losing hand *and* conditions suggest a bluff is viable.
- **Goal**: Decide whether to bluff in the current betting round.
- **Outputs**:
  - `{ "do_bluff": true|false }`
  - Potential bluff sizing guidelines.
- **Notes**:
  - Decision may be deferred – an agent may choose not to bluff now but keep bluffing open for a later street.

---

### 3. Board Strength Assessment
- **When triggered**: Post-flop, turn, or river.
- **Goal**: Evaluate how the community cards interact with potential opponent holdings.
- **Inputs**:
  - Current board cards
  - Action history for context
- **Outputs**:
  - A summary like: `board favors drawing hands`, `board strongly favors made straights/flushes`, `dry board`.
- **Usage**: Provides context for both value betting and bluffing decisions.

---

### 4. Logging & Rationale Capture
- **Goal**: Record intermediate assessments and the final action for training/evaluation.
- **Output Example**:
  ```json
  {
    "action": "raise",
    "amount": 12,
    "reasoning": "Opponent aggression high, board favors my range, bluff equity sufficient"
  }

### 5. Conditional Prompt Components (from feature_engineering.py)
- **Goal**: Get descriptive features re: current game state from engineered features, conditionally represent as text













