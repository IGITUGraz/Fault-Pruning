import time


class Timer:
    def __init__(self):
        self._startime = None
        self._endtime = None
        self.difftime = None

    def __enter__(self):
        self._startime = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._endtime = time.time()
        self.difftime = self._endtime - self._startime
