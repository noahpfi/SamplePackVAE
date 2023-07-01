import json


def load_params(filename):
    with open(filename) as file:
        params = json.load(file)
    return params
