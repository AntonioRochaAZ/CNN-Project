from functools import wraps
from typing import Callable
import time

# Decorators:
def timer(f: Callable):
    """timer decorator for counting a function's running time."""
    @wraps(f)   # Just for documentation purposes.
    def wrapper(*args, **kwargs):
        start = time.time()
        out = f(*args, **kwargs)
        delta_t = time.time() - start
        print(f"Time for running function {f.__name__}: {delta_t}s")
        return out
    return wrapper
