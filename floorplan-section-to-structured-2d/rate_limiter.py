import threading
import time


__all__ = ["RateLimiter"]

class RateLimiter:
    def __init__(self, requests_per_second):
        self.interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()

            if now < self.next_allowed:
                sleep_for = self.next_allowed - now
            else:
                sleep_for = 0.0

            self.next_allowed = max(
                now,
                self.next_allowed
            ) + self.interval

        if sleep_for > 0:
            time.sleep(sleep_for)
