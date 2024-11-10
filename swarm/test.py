import argparse
import json
from .settings import load_settings
from .synchronous_worker import startup, do_work

def run_test(job_id, job, output_dir):
    settings = load_settings()
    startup()
    try:
        do_work(job_id, job, output_dir)
        print("ok")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a job with the specified job_id.")
    parser.add_argument("job_id", type=str, help="The ID of the job to run")
    args = parser.parse_args()

    job_id = args.job_id
    job = None
    with open('./examples.json', 'r') as file:
        data = json.load(file)
        job = data.pop(job_id, None)        
    
    if job is not None:
        run_test(job_id, job, "./outputs")
    else:
        print("Job not found " + job_id)