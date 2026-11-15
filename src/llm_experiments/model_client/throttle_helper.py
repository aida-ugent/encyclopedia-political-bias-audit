import time


class ThrottleManager:
    """
    Manages rate limiting/throttling for API calls based on tokens per minute.
    Stateful; intended for single-threaded use.
    """

    def __init__(self, tpm_limit: int, window: float = 60.0):
        """
        Args:
            tpm_limit: Maximum tokens per minute (TPM)
            window: Time window in seconds (default 60.0 for 1 minute)
        """
        self.tpm_limit = tpm_limit
        self.window = window
        self.used_tokens = 0
        self.window_start = time.time()

    def throttle(self, tokens: int):
        """
        Throttle based on token count. Will sleep if needed to stay within limits.

        Args:
            tokens: Number of tokens for this request
        """
        now = time.time()

        # Reset window if it has elapsed
        if now - self.window_start >= self.window:
            self.used_tokens = 0
            self.window_start = now

        # If adding these tokens would exceed limit, wait for window to reset
        if self.used_tokens + tokens > self.tpm_limit:
            sleep_time = self.window - (now - self.window_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.used_tokens = 0
            self.window_start = time.time()

        # Increment token usage
        self.used_tokens += tokens
