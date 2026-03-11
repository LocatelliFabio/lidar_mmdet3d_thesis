# thread_safe_buffer.py

from threading import Lock


class LatestValueBuffer:
    def __init__(self):
        self._value = None
        self._timestamp = None
        self._seq = -1
        self._lock = Lock()

    def set(self, value, timestamp: float):
        with self._lock:
            self._value = value
            self._timestamp = float(timestamp)
            self._seq += 1

    def get_latest_copy(self):
        with self._lock:
            if self._value is None:
                return None, None, self._seq
            return self._value.copy(), self._timestamp, self._seq