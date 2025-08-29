# AI Poker Agents

This project runs real-time poker games between AI models like Claude, GPT, Gemini, and DeepSeek. Each model plays Texas Hold’em with the same information a human would: hole cards, community cards, betting history, and opponent stats. Their actions are based on their own reasoning and strategy.


## Features
- **Agent decision-making**: Each model receives comprehensive game information and uses strategic reasoning to make decisions
- **Full poker mechanics**: Complete Texas Hold'em rules with side pot handling, position-aware strategy, and blind defense
- **Multi-hand tournaments**: Run extended matches and observe how different models' playing styles emerge over time

## Agent Decision Process

**Information Gathering**: The agent decides what analysis to run:
- **Opponent statistics**: VPIP, aggression patterns, fold rates from previous hands
- **Hand analysis**: Equity calculations, draw probabilities, hand strength evaluation

**Strategic Decision**: The agent evaluates multiple factors to determine its play:
- **Action type**: 
  - Agent considers: Game state (cards, pot, position, betting history), opponent statistics (VPIP/PFR/aggression), hand analysis (strength, draw probabilities), strategic context (pot odds, blind defense, board texture)
  - Agent decides: 
    ```
    fold | check | call | bet | raise | all_in
    ```
  - Agent considers bluffing: If initial decision is passive (check/call), evaluates switching to aggressive bluff based on board texture and opponent fold rates
- **Bet sizing**: 
  - Agent considers: Hand strength, pot size, stack depth, position, opponent calling ranges
  - Agent decides: Optimal size for value extraction or bluff effectiveness

**Reasoning**: The agent provides explanations for every decision, showing its strategic thinking and the factors that influenced each choice.



## Quick Start

1. **Set up your [OpenRouter](https://openrouter.ai/) API keys** in `.env`:
   ```
   OPEN_ROUTER_TOKEN=your_openrouter_api_key
   ```

2. **Configure players** in `config.yaml`:
   ```yaml
   players:
     - name: "Claude Sonnet 4"
       model: "anthropic/claude-sonnet-4"
       chips: 250
       enabled: true
     - name: "GPT-4"
       model: "openai/gpt-4o"
       chips: 250
       enabled: true
   ```

3. **Run the tournament**:
   ```bash
   python app.py
   ```

4. **Watch** at http://127.0.0.1:5000/

## Running Games  

You can run games in two ways:  
- **Manual mode**: Step through each action to see how models make decisions.  
- **Auto-play mode**: Let a full match run without interaction.  

Game settings (blinds, starting stacks, hand limits) are configurable in `config.yaml`.  

## Output  

During play you’ll see:  
- Each model’s action and its reasoning  
- Pot sizes, chip counts, and player stats (VPIP, PFR, aggression factor)  
- Showdowns with full hand evaluation and side pot resolution  

Multi-hand runs highlight differences in style between models (tight vs loose, aggressive vs passive).  



---
