from DataLocator import DataLocator
from FileCopy import copy

import json
import os

def loadConfig(configPath = "./config.json"):
    with open(configPath, 'r') as file:
        return json.load(file)


if __name__ == '__main__':
    config = loadConfig()
    
    for job in config["copyJobs"]:
        dl = DataLocator(job["source"])
        all_json_files = dl.get_files(".json")

        copy(all_json_files, os.path.join(job), job["batchSize"])
        