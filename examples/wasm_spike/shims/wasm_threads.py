"""WASM spike shim: no threads in Pyodide.

ThreadPoolExecutor -> runs each submit() inline and returns a done future.
threading.Thread.start -> records the thread and does NOT run it (a loop
thread run inline would never return). Every skipped thread is a finding.
"""
import concurrent.futures as cf, threading
SKIPPED = []
class InlineExecutor(cf.Executor):
    def __init__(self, *a, **k): pass
    def submit(self, fn, *a, **k):
        f = cf.Future()
        try: f.set_result(fn(*a, **k))
        except BaseException as e: f.set_exception(e)
        return f
    def shutdown(self, wait=True, *, cancel_futures=False): pass
cf.ThreadPoolExecutor = InlineExecutor
import concurrent.futures.thread as cft; cft.ThreadPoolExecutor = InlineExecutor
def _start(self):
    SKIPPED.append(f"{self.name} target={getattr(self._target,'__qualname__',self._target)}")
    self._started.set()
threading.Thread.start = _start
