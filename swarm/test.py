from . import __version__
from .gpu.device import Device
from .settings import load_settings
import json
from .settings import load_settings
from .synchronous_worker import startup, do_work


def run_test(job):
    settings = load_settings()
    startup()
    try:
        func, args = format_args(job, settings, "cuda")
        result = do_work(Device(0), func, args)

        if "error" in result["pipeline_config"]:
            print(result["pipeline_config"]["error"])
        else:
            print("ok")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    job = None
    job_name = "flux_fast"
    with open('./examples.json', 'r') as file:
        data = json.load(file)
        job = data.pop(job_name, None)        
    
    if job is not None:
        run_test(job)
    else:
        print("Job not found " + job_name)

    
def format_args(job, settings, device_identifier):
    return job["worker_function"], job["args"]