import aiohttp
import json
from datetime import datetime
from .settings import save_file
from . import __version__
import logging


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
                try:
                    response_dict = await response.json()
                    return response_dict["jobs"]

                except Exception as e:
                    logging.exception(e)
                    print(f" error parsing response {e}")
                    return []

            elif response.status == 400:
                # this is when workers are not returning results within expectations
                # in this case the hive will return a message with further details
                response_dict = await response.json()
                message = response_dict.pop("message", "bad worker")
                print(f"{hive_uri} says {message}")

            else:
                print(f"{hive_uri} returned {response.status}")

            response.raise_for_status()
            return []


async def submit_result(settings, hive_uri, result):
    print("Result complete")

    timeout = aiohttp.ClientTimeout(total=90)
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
            resultResponse.raise_for_status()
            response_dict = await resultResponse.json()
            print(f"Result {response_dict}")


async def get_models(hive_uri):
    print(f"Fetching known model list from the hive at {hive_uri}...")

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "user-agent": f"chiaSWARM.worker/{__version__}",
            }
            async with session.get(
                f"{hive_uri}api/models", timeout=10, headers=headers
            ) as response:
                data = await response.json()
                save_file(data, "models.json")

                print("done")
                return data["language_models"] + data["models"]

    except Exception as e:
        print(f"Failed to fetch known model list from {hive_uri}: {e}")
        return []
