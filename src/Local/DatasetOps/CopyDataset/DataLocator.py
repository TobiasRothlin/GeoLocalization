from concurrent.futures import ThreadPoolExecutor, as_completed

import os

from tqdm import tqdm

import json

from time import time

class DataLocator:

    def __init__(self, folder, number_of_threads=16):
        self.folder = folder
        self.number_of_threads = number_of_threads
  
        self.folder_all_files = []

        self.max_age = 60*60*24 # 24 hours
        self.cache_file_path = f"{self.folder}/cache.json"
        self.cache_content = {}

        self.did_update_cache = False

        if self.load_cache():
            if time() - self.last_cache_update > self.max_age:
                print("Cache is older than 24 hours")
                print("Indexing all files")
                self.indexAllFiles()
                self.save_cache()
            else:
                print("Cache is up to date")

        else:
            print("Indexing all files")
            self.indexAllFiles()
            self.save_cache()


    def load_cache(self):
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, "r") as f:
                self.cache_content = json.load(f)

            self.folder_all_files = self.cache_content["folder_all_files"]
            self.last_cache_update = self.cache_content["last_cache_update"]

            return True
        else:
            print("No cache file found")
            return False

    def save_cache(self):
        self.did_update_cache = True
        self.cache_content["folder_all_files"] = self.folder_all_files
        self.cache_content["last_cache_update"] = time()

        with open(self.cache_file_path, "w") as f:
            json.dump(self.cache_content, f, indent=4)

    def indexAllFiles(self):
        self.folder_all_files = self.__search_files_in_directory(self.folder)


    def get_files(self, extension):
        return [f for f in tqdm(self.folder_all_files, desc=f"Gathering files with extention {extension}") if f.endswith(extension)]
        


    def __search_files_in_subdirectory(self, subdirectory):
        files = []
        for root, _, filenames in os.walk(subdirectory):
            for filename in filenames:
                    files.append(os.path.join(root, filename))
        return files
    

    def __search_files_in_directory(self, directory):
        subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if
                        os.path.isdir(os.path.join(directory, d))]
        
        all_files = []
        with ThreadPoolExecutor(max_workers=self.number_of_threads) as executor:
            futures = {executor.submit(self.__search_files_in_subdirectory, subdirectory): subdirectory for subdirectory
                    in subdirectories}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Indexing directorys"):
                all_files.extend(future.result())
        return all_files
    


if __name__ == "__main__":
    dl_Test = DataLocator("/Volumes/DATASET/Test")
    print(len(dl_Test.get_files(".json")))
    print(len(dl_Test.get_files(".jpg")))
    print(len(dl_Test.get_files(".jpeg")))

    dl_Train = DataLocator("/Volumes/DATASET/Train")
    print(len(dl_Train.get_files(".json")))
    print(len(dl_Train.get_files(".jpg")))
    print(len(dl_Train.get_files(".jpeg")))