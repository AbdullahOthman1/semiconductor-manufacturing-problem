#!/usr/bin/python
import math
import json

DEFAULT_PATH = "../results/"

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def save_json_file(file_name, json_object):
    with open(f"{DEFAULT_PATH}/{file_name}.json", 'w') as outfile:
        json.dump(json_object, outfile)