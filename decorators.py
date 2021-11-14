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
        hours = delta_t//3600
        delta_t = delta_t % 3600
        minutes = delta_t // 60
        seconds = delta_t % 60
        print(f"Time for running function {f.__name__}:\n"
              f"{hours} hours, {minutes} minutes and {seconds} seconds")
        return out
    return wrapper
