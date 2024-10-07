import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from DataLocator import DataLocator

from ParseFiles import getLatLongFiles,getCountriesFiles
from GetCoordinates import getCoordinates

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

LIST_OF_COUNTRIES_PATH = "/home/tobias.rothlin/GeoLocalization/src/DGX1/src/FindeSparseRegions/ListOfCountries.txt"

if __name__ == "__main__":

    print("\033[37m") # Light Gray
    
    dl_Test = DataLocator(TEST_DATA_FOLDER)
    dl_Train = DataLocator(TRAIN_DATA_FOLDER)

    test_files_json = dl_Test.get_files(".json")

    train_files_json = dl_Train.get_files(".json")

    print("\033[0m") # Reset Color

    train_locations = getLatLongFiles(train_files_json)
    test_locations = getLatLongFiles(test_files_json)

    train_countries = getCountriesFiles(train_files_json)
    test_countries = getCountriesFiles(test_files_json)

    list_of_all_countries = []
    with open(LIST_OF_COUNTRIES_PATH, "r") as f:
        for line in f:
            list_of_all_countries.append(line.strip().split(";")[0])

    set_of_all_countries_in_train = {country.lower() for country in train_countries}
    set_of_all_countries_in_test = set(country.lower() for country in test_countries)

    missing_countries_in_train = {country for country in list_of_all_countries if country.lower() not in set_of_all_countries_in_train}
    missing_countries_in_test = {country for country in list_of_all_countries if country.lower() not in set_of_all_countries_in_test}

    print(f"Missing countries in Train {len(missing_countries_in_train)}/{len(list_of_all_countries)}({100*len(missing_countries_in_train)/len(list_of_all_countries)})%")
    print(f"Missing countries in Test {len(missing_countries_in_test)}/{len(list_of_all_countries)}({100*len(missing_countries_in_test)/len(list_of_all_countries)})%")


    with open("missing_countries_in_train.txt", "w") as f:
        for country in missing_countries_in_train:
            f.write(f"{country}\n")

    with open("missing_countries_in_test.txt", "w") as f:
        for country in missing_countries_in_test:
            f.write(f"{country}\n")


    # Get Coordinates of missing countries
    missing_countries_in_train_coordinates = {}
    missing_countries_in_test_coordinates = {}

    for country in missing_countries_in_train:
        missing_countries_in_train_coordinates[country] = getCoordinates(country)

    for country in missing_countries_in_test:
        missing_countries_in_test_coordinates[country] = getCoordinates(country)

    


