# thread_safe_buffer.py

from threading import Lock


class LatestValueBuffer:
    def __init__(self):
        self._value = None
        self._lock = Lock()

    def set(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value

    def get_copy(self):
        with self._lock:
            if self._value is None:
                return None
            return self._value.copy()