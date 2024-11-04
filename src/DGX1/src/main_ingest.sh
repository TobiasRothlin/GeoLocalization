#!/bin/bash

#If -r option is used, the script will run the srips with the -r option
REBUILD=false
CUDA_VISIBLE_DEVICES=""


# Path to the scripts
SCRIPT_FILE_VALIDATION=/home/tobias.rothlin/GeoLocalization/src/DGX1/src/FileValidation/FileValidation.sh
SCRIPT_REVERSE_GEOCODING=/home/tobias.rothlin/GeoLocalization/src/DGX1/src/ReverseGeocoding/ReverseGeocoding.sh
SCRIPT_MAPILLARY_COLLECTION=/home/tobias.rothlin/GeoLocalization/src/DGX1/src/MapillaryCollection/MapillaryCollection.sh
SCRIPT_CLASSIFICATION=/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Classification/ImageClassificationPipeline.sh
SCRIPT_VISUALIZATION=/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Visualization/Visualization.sh

# Parse the command line arguments
while getopts "rg:" opt; do
  case $opt in
    r)
      REBUILD=true
      ;;
    g)
      CUDA_VISIBLE_DEVICES=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Ensure each script is executable
chmod +x $SCRIPT_FILE_VALIDATION
chmod +x $SCRIPT_REVERSE_GEOCODING
chmod +x $SCRIPT_CLASSIFICATION



# Run each script in the specified order
#-------------------------------------------------------------------------------------------------

if [ "$REBUILD" = true ]; then
    echo "Building and Running Script $SCRIPT_REVERSE_GEOCODING"
    $SCRIPT_REVERSE_GEOCODING -r
    if [ $? -ne 0 ]; then
        echo "$SCRIPT_REVERSE_GEOCODING failed"
        exit 1
    fi
else
    echo "Running Script $SCRIPT_REVERSE_GEOCODING"
    $SCRIPT_REVERSE_GEOCODING
    if [ $? -ne 0 ]; then
        echo "$SCRIPT_REVERSE_GEOCODING failed"
        exit 1
    fi
fi

#-------------------------------------------------------------------------------------------------
if [ "$REBUILD" = true ]; then
    echo "Building and Running Script $SCRIPT_CLASSIFICATION"
    $SCRIPT_CLASSIFICATION -r -g $CUDA_VISIBLE_DEVICES
    if [ $? -ne 0 ]; then
        echo "$SCRIPT_CLASSIFICATION failed"
        exit 1
    fi
else
    echo "Running Script $SCRIPT_CLASSIFICATION"
    $SCRIPT_CLASSIFICATION -g $CUDA_VISIBLE_DEVICES
    if [ $? -ne 0 ]; then
        echo "$SCRIPT_CLASSIFICATION failed"
        exit 1
    fi
fi


#-------------------------------------------------------------------------------------------------

if [ "$REBUILD" = true ]; then
    echo "Building and Running Script $SCRIPT_FILE_VALIDATION"
    $SCRIPT_FILE_VALIDATION -r
    if [ $? -ne 0 ]; then
        echo "$SCRIPT_FILE_VALIDATION failed"
        exit 1
    fi
else
    echo "Running Script $SCRIPT_FILE_VALIDATION"
    $SCRIPT_FILE_VALIDATION
    if [ $? -ne 0 ]; then
        echo "$SCRIPT_FILE_VALIDATION failed"
        exit 1
    fi
fi

#-------------------------------------------------------------------------------------------------

if [ "$REBUILD" = true ]; then
    echo "Building and Running Script $SCRIPT_VISUALIZATION"
    $SCRIPT_VISUALIZATION -r
    if [ $? -ne 0 ]; then
        echo "$SCRIPT_VISUALIZATION failed"
        exit 1
    fi
else
    echo "Running Script $SCRIPT_VISUALIZATION"
    $SCRIPT_VISUALIZATION
    if [ $? -ne 0 ]; then
        echo "$SCRIPT_VISUALIZATION failed"
        exit 1
    fi
fi

echo "All scripts ran successfully"