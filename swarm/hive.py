import aiohttp
import json
from datetime import datetime
from . import __version__


async def ask_for_work(settings, hive_uri):
    print(f"{datetime.now()}: Asking for work from the hive at {hive_uri}...")

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
            f"{hive_uri}/work",
            timeout=timeout.total,
            params={
                "worker_version": __version__,
                "worker_name": settings.worker_name,
            },
            headers={
                "Content-type": "application/json",
                "Authorization": f"Bearer {settings.sdaas_token}",
                "user-agent": f"chiaSWARM.worker/{__version__}",
            },
        ) as response:
            if response.status == 200:
                response_dict = await response.json()
                return response_dict["jobs"]

            elif response.status == 400:
                # this is when workers are not returning results within expectations
                response_dict = await response.json()
                message = response_dict.pop("message", "bad worker")
                print(f"{hive_uri} says {message}")
                response.raise_for_status()

            else:
                print(f"{hive_uri} returned {response.status}")
                response.raise_for_status()


async def submit_result(settings, hive_uri, result):
    print("Result complete")

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{hive_uri}/results",
            data=json.dumps(result),
            headers={
                "Content-type": "application/json",
                "Authorization": f"Bearer {settings.sdaas_token}",
                "user-agent": f"chiaSWARM.worker/{__version__}",
            },
        ) as resultResponse:
            if resultResponse.status == 500:
                print(f"The hive returned an error: {resultResponse.reason}")
            else:
                response_dict = await resultResponse.json()
                print(f"Result {response_dict}")
