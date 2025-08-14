#!/usr/bin/python

from datetime import datetime
import psutil
import sys
import socket
from utils import convert_size, save_json_file
import logging
from typing import Any, Dict
    
LOG = logging.getLogger(__name__)

DEFAULT_FREQUENCY = 1
DEFAULT_DURATION = 60
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
ASK_TO_CONTINUIE = True



def run_interval(duration: int, frequency = int):
    counts = int(duration / frequency)
    for x in range(counts):
        dt = datetime.now()
        timestamp = datetime.timestamp(dt)
        datetime_formmated = dt.now().strftime(DATETIME_FORMAT)
        cpu_percent = psutil.cpu_percent(interval=frequency)
        mem = psutil.virtual_memory()
        print(f"{datetime_formmated} The percentage of CPU utilization  {cpu_percent}%")
        print(f"{datetime_formmated} Total RAM: {convert_size(mem.total)}, Available RAM:  {convert_size(mem.available)}, Used RAM: {convert_size(mem.used)}")
        print(f"{datetime_formmated} The percentage of RAM utilization  {mem.percent}%")

        cpu_record = {"timestamp": timestamp, "percent": cpu_percent}
        ram_record = {"timestamp": timestamp, "percent": mem.percent, "available": mem.available, "used": mem.used}
        cpu_utilization_json.append(cpu_record)
        ram_utilization_json.append(ram_record)


def build_json_body(
    id: str,
    title: str,
    hostname: str,
    start_time: str,
    total_duration: int,
    frequency: int,
    resources: Dict[str, Any],
    cpu_utilization_json: Dict[str, Any],
    ram_utilization_json: Dict[str, Any]
    ) -> Dict[str, Any]:

    json_output = {
        "id": id,
        "title": title,
        "host": hostname,
        "start_time": start_time,
        "duration": total_duration,
        "frequency": frequency,
        "resources": resources,
        "utilization": {
            "cpu": cpu_utilization_json,
            "ram": ram_utilization_json
        }
    }

    return json_output

if __name__ == "__main__":
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    id = sys.argv[1] if len(sys.argv) >= 2 else "1"
    frequency = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_FREQUENCY
    duration = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_DURATION
    title = sys.argv[4] if len(sys.argv) >= 5 else "Stress Test"

    total_duration = duration
    datetime_object = datetime.now()
    start_time = datetime_object.now().strftime(DATETIME_FORMAT)
    hostname = socket.gethostname()

    print(f"id = {id}")
    print(f"title = {title}")
    print(f"frequency = {frequency} seconds")
    print(f"duration = {duration} seconds")

    cpu_utilization_json = []
    ram_utilization_json = []
    resources_json = {
        "cpu": {
            "logical_cpus": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False)
        },
        "ram": {
            "total": convert_size(psutil.virtual_memory().total)
        }
    }

    print(f"The number of logical CPUs in the system {psutil.cpu_count()}")
    print(f"The number of physical cores in the system {psutil.cpu_count(logical=False)}")

    file_name = f"stress_test_id[{id}]_{hostname}_[{start_time}]"

    continue_run = True
    
    while continue_run:
        run_interval(duration, frequency)
        json_output = build_json_body(id=id, title=title, hostname=hostname, start_time=start_time,
         total_duration=total_duration, frequency=frequency, resources=resources_json,
         cpu_utilization_json=cpu_utilization_json, ram_utilization_json=ram_utilization_json)

        save_json_file(file_name, json_output)
        if ASK_TO_CONTINUIE:
            user_continue = input("Do you like to continue (True/False)?")
            continue_run = (user_continue == "True")
            if continue_run:
                new_duration = int(input("Enter new duration in seconds: "))
                duration = new_duration
                total_duration = total_duration + duration
            else: continue_run = False
        else: continue_run = False

    

    
