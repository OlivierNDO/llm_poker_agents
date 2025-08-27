# main.py
import time
from src.logging_config import logger
from src.game import TableConfig, Player, StatsTracker, TableManager  # game state & runtime
from src.agents import LLMAgent                           # agents
from src.agent_utils import OpenRouterCompletionEngine    # LLM client adapter



# =============================================================================
# POKER GAME SETUP
# =============================================================================
_log_prompts = False
_n_chips = 250
config = TableConfig(small_blind=1, big_blind=2, max_hands=1)
stats = StatsTracker()

# =============================================================================
# LLM CLIENT SETUP
# =============================================================================
open_ai_gpt_4o_client = OpenRouterCompletionEngine(
    model='openai/gpt-4o'
)

anthropic_claude_sonnet_4_client = OpenRouterCompletionEngine(
    model='anthropic/claude-sonnet-4'
)

google_gemini_flash_2_5_client = OpenRouterCompletionEngine(
    model='google/gemini-2.5-flash'
)

llama_3_3_70B_client = OpenRouterCompletionEngine(
    model='meta-llama/llama-3.3-70b-instruct'
)

# =============================================================================
# PLAYER CREATION
# =============================================================================
player_open_ai_gpt_4o = Player(
    name='Open AI GPT-4o',
    chips=_n_chips,
    agent=LLMAgent(
        client=open_ai_gpt_4o_client,
        name='Open AI GPT-4o',
        log_prompts=_log_prompts,
        table_config=config
    )
)

player_claude_sonnet_4 = Player(
    name='Claude Sonnet 4',
    chips=_n_chips,
    agent=LLMAgent(
        client=anthropic_claude_sonnet_4_client,
        name='Claude Sonnet 4',
        log_prompts=_log_prompts,
        table_config=config
    )
)

player_gemini_flash_2_5 = Player(
    name='Gemini Flash 2.5',
    chips=_n_chips,
    agent=LLMAgent(
        client=google_gemini_flash_2_5_client,
        name='Gemini Flash 2.5',
        log_prompts=_log_prompts,
        table_config=config
    )
)

player_llama_3_3_70B = Player(
    name='Llama 3.3 70B',
    chips=_n_chips,
    agent=LLMAgent(
        client=llama_3_3_70B_client,
        name='Llama 3.3 70B',
        log_prompts=_log_prompts,
        table_config=config
    )
)

# Active players for this session
players = [
    player_open_ai_gpt_4o,
    player_gemini_flash_2_5,
    player_llama_3_3_70B
]

# =============================================================================
# GAME LOOP
# =============================================================================
def main():
    # Create the table manager
    tm = TableManager(players, config, stats)
    
    # Run hands until stop condition (bust-out or max_hands)
    tm.play()
    
    # Final results
    logger.info("=== FINAL RESULTS ===")
    for p in players:
        logger.info(f'{p.name}: {p.chips} chips')

if __name__ == "__main__":
    main()