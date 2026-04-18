import time
import asyncio
from functools import wraps

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

def retry_with_backoff(retries:int):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    print(f"Error {e}. Retrying immediately...")
                    # Immediate retry, no sleep needed
                    x += 1

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    print(f"Error {e}. Retrying immediately...")
                    # Immediate retry, no sleep needed
                    x += 1
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator