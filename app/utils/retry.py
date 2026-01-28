import asyncio


async def retry_async(fn, retries=3, timeout=15):
    last_error = None

    for _ in range(retries):
        try:
            return await asyncio.wait_for(fn(), timeout=timeout)
        except Exception as e:
            last_error = e

    raise last_error
