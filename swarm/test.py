from . import __version__
from .settings import load_settings
import json
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
    job = None
    job_id = "kandinsky_3_img2img"
    with open('./examples.json', 'r') as file:
        data = json.load(file)
        job = data.pop(job_id, None)        
    
    if job is not None:
        run_test(job_id, job, "./outputs")
    else:
        print("Job not found " + job_id)
