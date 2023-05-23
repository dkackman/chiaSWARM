from typing import Union
import json
import os
from pathlib import Path


class Settings:
    # when true huggingface will look for auth from the environment -
    # otherwise the api key itself
    huggingface_token: Union[bool, str] = True
    log_level: str = "WARN"
    log_filename: str = "log/generator.log"
    sdaas_token: str = ""
    sdaas_uri: str = "http://localhost:9511"
    worker_name: str = "worker"


def load_settings():
    settings = Settings()
    try:
        with open(get_settings_full_path(), "r") as file:
            settings_dict = json.load(file)
    except FileNotFoundError:
        print("no settings file")
        settings_dict = {}
    except json.JSONDecodeError:
        print("invalid settings file")
        settings_dict = {}

    settings.log_level = settings_dict.get("log_level", "WARN")
    settings.log_filename = settings_dict.get("log_filename", "log/generator.log")
    settings.sdaas_token = settings_dict.get("sdaas_token", "")
    settings.sdaas_uri = settings_dict.get("sdaas_uri", "http://localhost:9511")
    settings.worker_name = settings_dict.get("worker_name", "worker")

    settings.sdaas_token = os.getenv("SDAAS_TOKEN", settings.sdaas_token)
    settings.sdaas_uri = os.getenv("SDAAS_URI", settings.sdaas_uri)
    settings.worker_name = os.getenv("SDAAS_WORKERNAME", settings.worker_name)

    return settings


def save_settings(settings):
    settings_dict = settings.__dict__
    with open(get_settings_full_path(), "w") as file:
        json.dump(settings_dict, file, indent=2)


def settings_exist():
    return get_settings_full_path().is_file()


def resolve_path(path):
    full_path = get_settings_dir().joinpath(path)
    # make the directory if it doesn't exist
    full_path.parent.mkdir(parents=True, exist_ok=True)

    return full_path


def get_settings_dir():
    dir_path = os.environ.get("SDAAS_ROOT")
    if not dir_path:
        dir_path = "~/.sdaas/"

    return Path(dir_path).expanduser()


def save_file(data, filename):
    with open(resolve_path(filename), "w") as file:
        json.dump(data, file, indent=2)


def get_settings_full_path():
    return resolve_path("settings.json")
