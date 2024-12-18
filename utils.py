# FILE: utils.py
import time
import asyncio
from functools import wraps
import logging

def measure_time(description):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            print(f"⏱️ {description}: {end_time - start_time:.2f} seconds")
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"⏱️ {description}: {end_time - start_time:.2f} seconds")
            return result
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator