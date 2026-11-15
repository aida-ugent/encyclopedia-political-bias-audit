import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "batch_size": 50,
    "requests_per_minute": 100,
    "min_delay_between_requests": 0.5,
    "max_retries": 5,
    "initial_backoff": 1.0,
    "max_backoff": 60.0,
    "backoff_multiplier": 2.0,
    "batch_delay": 1.0,
    "checkpoint_frequency": 10,
    "max_workers": 5,  # Parallel workers (keep low to avoid API rate limits)
}

STOP_HEADINGS = ["See also", "References", "External links", "Further reading", "Notes", "Bibliography"]
