# AI Poker Agents

This project runs real-time poker games between AI models like Claude, GPT, Gemini, and DeepSeek. Each model plays Texas Hold'em with the same information a human would: hole cards, community cards, betting history, and opponent stats. Their actions are based on their own reasoning and strategy.


## Features
- **Agent decision-making**: Each model receives comprehensive game information and uses strategic reasoning to make decisions
- **Full poker mechanics**: Complete Texas Hold'em rules with side pot handling, position-aware strategy, and blind defense
- **Multi-hand tournaments**: Run extended matches and observe how different models' playing styles emerge over time


## Agent Decision Process

### Information Gathering
The agent decides what analysis to run:
- **Opponent statistics**: VPIP, aggression patterns, fold rates from previous hands
- **Hand analysis**: Equity calculations, draw probabilities, hand strength evaluation

### Strategic Decision
The agent evaluates multiple factors:
- **Action type**: Considers game state, opponent stats, hand strength → decides fold/check/call/bet/raise/all_in
- **Bet sizing**: Considers hand strength, pot size, position → decides optimal sizing
- **Bluffing**: If initial decision is passive, evaluates aggressive bluff based on board texture

### Reasoning
The agent provides explanations for every decision, showing strategic thinking and influencing factors.


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
   python mcp_app.py
   ```

4. **Watch** at http://127.0.0.1:5000/

## Running Games  

You can run games in two ways:  
- **Manual mode**: Step through each action to see how models make decisions.  
- **Auto-play mode**: Let a full match run without interaction.  

Game settings (blinds, starting stacks, hand limits) are configurable in `config.yaml`.  

## Architecture

### Core Components

- `mcp_app.py`: Flask application and game coordinator
- `src/game.py`: Game state management and hand execution
- `src/agents.py`: AI agent implementations
- `src/feature_engineering.py`: Hand evaluation and probability calculations
- `templates/poker_table.html`: Web interface

### Key Classes

- `PokerGameServer`: Main application controller
- `TableManager`: Handles dealer rotation and game flow
- `Hand`: Individual hand state and betting round execution
- `LLMAgent`: Basic agent using LLM completions
- `LLMAgentMCP`: Enhanced agent with MCP tool access


## License
MIT

## Contact
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/oliviernicholas/)
