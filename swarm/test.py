from . import __version__
from .settings import load_settings
import json
from .pipeline_processors.arguments import prepare_args
from .synchronous_worker import startup, do_work


def run_test(job, output_dir):
    settings = load_settings()
    startup()
    try:
        args = prepare_args(job)
        result = do_work(args, output_dir)

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
        run_test(job, "./outputs")
    else:
        print("Job not found " + job_name)
