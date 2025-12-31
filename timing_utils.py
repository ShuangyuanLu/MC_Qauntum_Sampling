import time
from functools import wraps

def timecall(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        print(f"{fn.__name__} took {time.perf_counter() - t0:.3f}s")
        return out
    return wrapper