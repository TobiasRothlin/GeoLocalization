from concurrent.futures import ThreadPoolExecutor, as_completed

import os

from tqdm import tqdm

class DataLocator:

    def __init__(self, folder, number_of_threads=16):
        self.folder = folder
        self.number_of_threads = number_of_threads
  
        self.folder_all_files = []

        self.indexAllFiles()


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