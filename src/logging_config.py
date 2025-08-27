"""
logging_config.py

Global logger configuration - import 'logger' directly from this module
"""
import logging
import os
from datetime import datetime

# Create logs directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"poker_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Create and export a global logger
logger = logging.getLogger("poker_game")

# Keep the get_logger function for backwards compatibility
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)