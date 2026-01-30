import time

#this is for preventing system from repeatedly calling a failing external service.
class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.open = False

    def record_success(self):
        self.failure_count = 0
        self.open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.open = True

    def allow_request(self):
        if not self.open:
            return True

        if time.time() - self.last_failure_time > self.recovery_timeout:
            self.open = False
            self.failure_count = 0
            return True

        return False
