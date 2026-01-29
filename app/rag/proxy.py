# import os
# for k, v in os.environ.items():
#     if isinstance(v, dict):
#         print("BAD ENV VAR:", k, v)
# import asyncio
# import httpx

# async def main():
#     async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
#         r = await client.post(
#             "http://127.0.0.1:11434/api/generate",
#             json={"model": "llama3", "prompt": "hello", "stream": False}
#         )
#         print(r.status_code)
#         print(r.text)

# asyncio.run(main())
