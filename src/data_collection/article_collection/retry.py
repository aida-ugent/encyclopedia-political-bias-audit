import time
from functools import wraps
from mwapi.errors import APIError
from config import CONFIG, logger


def retry_with_backoff(max_retries: int = None, initial_backoff: float = None):
    """Decorator for retry logic with exponential backoff."""
    max_retries = max_retries or CONFIG["max_retries"]
    initial_backoff = initial_backoff or CONFIG["initial_backoff"]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            backoff = initial_backoff
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except APIError as e:
                    last_exception = e
                    error_code = str(e)

                    if "ratelimited" in error_code.lower() or "429" in error_code:
                        wait_time = min(backoff * 2, CONFIG["max_backoff"])
                        logger.warning(f"Rate limited! Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    else:
                        wait_time = min(backoff, CONFIG["max_backoff"])
                        logger.warning(f"API error: {e}. Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s")

                    if attempt < max_retries:
                        time.sleep(wait_time)
                        backoff *= CONFIG["backoff_multiplier"]

                except Exception as e:
                    last_exception = e
                    logger.error(f"Unexpected error: {e}. Retry {attempt + 1}/{max_retries}")
                    if attempt < max_retries:
                        time.sleep(backoff)
                        backoff *= CONFIG["backoff_multiplier"]

            logger.error(f"All {max_retries} retries exhausted")
            raise last_exception if last_exception else RuntimeError("Unknown error")

        return wrapper
    return decorator
