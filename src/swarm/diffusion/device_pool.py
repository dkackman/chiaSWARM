from typing import List
from .device import Device
from threading import Lock

mutex: Lock = Lock()

available: List[Device] = []


def get_available_gpu_count():
    mutex.acquire(True, 1)
    try:
        return len(available)
    finally:
        mutex.release()


def add_device_to_pool(device: Device):
    mutex.acquire(True, 2)
    try:
        available.append(device)
    finally:
        mutex.release()


def remove_device_from_pool() -> Device:
    mutex.acquire(True, 2)
    try:
        if len(available) > 0:
            return available.pop(0)

        raise Exception("busy")
    finally:
        mutex.release()
