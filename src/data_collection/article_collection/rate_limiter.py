import time
import threading
from typing import List
from config import CONFIG, logger


class RateLimiter:
    """Thread-safe rate limiter with sliding window for API requests."""

    def __init__(self, requests_per_minute: int = None):
        self.requests_per_minute = requests_per_minute or CONFIG["requests_per_minute"]
        self.request_times: List[float] = []
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits (thread-safe)."""
        with self._lock:
            current_time = time.time()

            # Clean old request times (older than 1 minute)
            self.request_times = [t for t in self.request_times if current_time - t < 60]

            # If we've hit the rate limit, wait
            if len(self.request_times) >= self.requests_per_minute:
                oldest_request = self.request_times[0]
                wait_time = 60 - (current_time - oldest_request) + 0.1
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                    # Release lock while sleeping so other threads can proceed
                    self._lock.release()
                    try:
                        time.sleep(wait_time)
                    finally:
                        self._lock.acquire()

            # Record this request
            self.request_times.append(time.time())


# Global rate limiter instance
rate_limiter = RateLimiter()
