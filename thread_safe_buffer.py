# thread_safe_buffer.py

from threading import Lock
from time import monotonic


class LatestValueBuffer:
    def __init__(self):
        self._value = None
        self._seq = 0
        self._timestamp = None
        self._lock = Lock()

    def set(self, value):
        with self._lock:
            self._value = value
            self._seq += 1
            self._timestamp = monotonic()

    def get(self):
        with self._lock:
            return self._value, self._seq, self._timestamp

    def get_copy(self):
        with self._lock:
            if self._value is None:
                return None, self._seq, self._timestamp
            return self._value.copy(), self._seq, self._timestamp