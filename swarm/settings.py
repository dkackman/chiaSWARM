from typing import Union
import json
import os
from pathlib import Path


def load_settings():
    dict = {}
    try:
        with open(get_settings_full_path(), "r") as file:
            dict = json.loads(file.read())
    except:
        print("no settings file")

    settings = Settings()
    # settings.huggingface_token = dict.get("huggingface_token", True)
    settings.log_level = dict.get("log_level", "WARN")
    settings.log_filename = dict.get("log_filename", "log/generator.log")
    settings.sdaas_token = dict.get("sdaas_token", "")
    settings.sdaas_uri = dict.get("sdaas_uri", "http://localhost:9511")
    settings.worker_name = dict.get("worker_name", "worker")

    # override settings file with environment vairables
    # if not os.environ.get("HUGGINGFACE_TOKEN", None) is None:
    #    settings.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN", "")

    if not os.environ.get("SDAAS_TOKEN", None) is None:
        settings.sdaas_token = os.environ.get("SDAAS_TOKEN", "")

    if not os.environ.get("SDAAS_URI", None) is None:
        settings.sdaas_uri = os.environ.get("SDAAS_URI", "")

    if not os.environ.get("SDAAS_WORKERNAME", None) is None:
        settings.worker_name = os.environ.get("SDAAS_WORKERNAME", "")

    return settings


def save_settings(settings):
    dir = Path(get_settings_dir())
    dir.mkdir(0o770, parents=True, exist_ok=True)

    with open(get_settings_full_path(), "w") as file:
        file.write(json.dumps(settings.__dict__, indent=2))


def settings_exist():
    return get_settings_full_path().is_file()


def resolve_path(path):
    path = get_settings_dir().joinpath(path)
    # make the directory if it doesn't exist
    path.parent.mkdir(0o770, parents=True, exist_ok=True)

    return path


def get_settings_dir():
    dir = os.environ.get("SDAAS_ROOT", None)
    if dir is None:
        dir = "~/.sdaas/"

    return Path(dir).expanduser()


def get_settings_full_path():
    return Path(get_settings_dir()).joinpath("settings.json")


class Settings:
    # when true huggingface will look for auth from the environment - otherwise the api key itself
    huggingface_token: Union[bool, str] = True
    log_level: str = "WARN"
    log_filename: str = "log/generator.log"
    sdaas_token: str = ""
    sdaas_uri: str = ""
    worker_name: str = "worker"
