import os

from time import sleep

from DataLocator import DataLocator
from CreateLableFile import createLableFile
from ReverseGeoCode import put_locations_to_json_files
from CheckJsonFileStatus import check_json_files

from tqdm import tqdm

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

if __name__ == "__main__":

    print("\033[37m") # Light Gray
    
    dl_Test = DataLocator(TEST_DATA_FOLDER)
    dl_Train = DataLocator(TRAIN_DATA_FOLDER)

    test_files_json = dl_Test.get_files(".json")
    test_files_jpg = dl_Test.get_files(".jpg")
    test_files_jpeg = dl_Test.get_files(".jpeg")

    train_files_json = dl_Train.get_files(".json")
    train_files_jpg = dl_Train.get_files(".jpg")
    train_files_jpeg = dl_Train.get_files(".jpeg")

    print("\033[0m") # Reset Color

    print(f"Checking if Meta Data files are present")
    test_files_json_meta = [f for f in test_files_json if "._" in f]
    test_files_jpg_meta = [f for f in test_files_jpg if "._" in f]
    test_files_jpeg_meta = [f for f in test_files_jpeg if "._" in f]

    train_files_json_meta = [f for f in train_files_json if "._" in f]
    train_files_jpg_meta = [f for f in train_files_jpg if "._" in f]
    train_files_jpeg_meta = [f for f in train_files_jpeg if "._" in f]

    if len(test_files_json_meta) == 0 and len(test_files_jpg_meta) == 0 and len(test_files_jpeg_meta) == 0:
        print("✅ \033[32m No Meta Data files found in Test folder \033[0m")

    else:
        print(f"❌ \033[31m {len(test_files_json_meta)+len(test_files_jpg_meta)+len(test_files_jpeg_meta)} Meta Data files found in Test folder \033[0m")

        if input("Do you want to remove these files? (y/n): ").lower() == "y":
            for f in test_files_json_meta + test_files_jpg_meta + test_files_jpeg_meta:
                os.remove(f)
                print(f"    -Removed {f}")


    if len(train_files_json_meta) == 0 and len(train_files_jpg_meta) == 0 and len(train_files_jpeg_meta) == 0:
        print("✅ \033[32m No Meta Data files found in Train folder \033[0m")
    
    else:
        print(f"❌ \033[31m {len(train_files_json_meta)+len(train_files_jpg_meta)+len(train_files_jpeg_meta)} Meta Data files found in Train folder \033[0m")

        if input("Do you want to remove these files? (y/n): ").lower() == "y":
            for f in train_files_json_meta + train_files_jpg_meta + train_files_jpeg_meta:
                os.remove(f)
                print(f"    -Removed {f}")

    

    print("\033[0m") # Reset Color

    print(f"\033[1m\u001b[32;1mTrain:\033[0m") # Bold and Green
    print(f"    JSON: {len(train_files_json):,d}")
    print(f"    JPG : {len(train_files_jpg):,d}")
    print(f"    JPEG: {len(train_files_jpeg):,d}")
    print(f"        Total: {len(train_files_jpg) +len(train_files_jpeg):,d}")

    if len(train_files_json) == len(train_files_jpg) +len(train_files_jpeg):
        print("✅ \033[32m All JSON and Image files are present in the Train folder \033[0m") # Green
    else:
        print(f"❌ \033[31m {len(train_files_json)-(len(train_files_jpg) +len(train_files_jpeg))} are missing ! \033[0m")   # Red

    print("\033[37m")   # Yellow
    print(100*"-")
    print("\033[0m")    # Reset Color



    print(f"\033[1m\u001b[34;1mTest:\033[0m")   # Bold
    print(f"    JSON: {len(test_files_json):,d}")
    print(f"    JPG : {len(test_files_jpg):,d}")
    print(f"    JPEG: {len(test_files_jpeg):,d}")
    print(f"        Total: {len(test_files_jpg) +len(test_files_jpeg):,d}")

    if len(test_files_json) == len(test_files_jpg) +len(test_files_jpeg):
        print("✅ \033[32m All JSON and Image files are present in the Test folder \033[0m")    # Green
    else:
        print(f"❌ \033[31m {len(test_files_json)-(len(test_files_jpg) +len(test_files_jpeg))} are missing ! \033[0m")  # Red 

    check_json_files(train_files_json + test_files_json, num_threads=32)

    put_locations_to_json_files(train_files_json + test_files_json, num_threads=32)

    check_json_files(train_files_json + test_files_json, num_threads=32)
    
    if dl_Test.did_update_cache:
        createLableFile(test_files_json, os.path.join(TEST_DATA_FOLDER, "test.csv"))

    if dl_Train.did_update_cache:
        createLableFile(train_files_json, os.path.join(TRAIN_DATA_FOLDER, "train.csv"))

    