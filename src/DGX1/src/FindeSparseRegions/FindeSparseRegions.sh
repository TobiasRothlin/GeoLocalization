#!/bin/bash
SIF_FOLDER="/home/tobias.rothlin/data/SIF"
IMAGE_NAME="$SIF_FOLDER/FindeSparseRegions.sif"
DEFINITION_FILE="/home/tobias.rothlin/GeoLocalization/src/DGX1/src/FindeSparseRegions/FindeSparseRegions.def"
SESSION_NAME="FindeSparseRegions"
REBUILD=false

APPTAINER_LOG_PATH="/home/tobias.rothlin/.apptainer/instances/logs/sifs-dgx/tobias.rothlin/FindeSparseRegions.err"

# Parse command-line options
while getopts "r" opt; do
  case $opt in
    r)
      REBUILD=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Build the Apptainer container if rebuild is necessary
if [ "$REBUILD" = true ]; then
  echo "Building the Apptainer container..."
  apptainer build $IMAGE_NAME $DEFINITION_FILE

  # Check if the build was successful
  if [ $? -ne 0 ]; then
      echo "Failed to build the Apptainer container."
      exit 1
  fi
fi

apptainer instance start $IMAGE_NAME $SESSION_NAME

sleep 15

# Function to find Jupyter server port and token
find_jupyter_info() {
  echo "Finding Jupyter server port and token..."
  if [ -f "$APPTAINER_LOG_PATH" ]; then
    PORT=$(grep -oP '(?<=http://localhost:)\d+' "$APPTAINER_LOG_PATH" | head -n 1)
    TOKEN=$(grep -oP '(?<=token=)[a-z0-9]+' "$APPTAINER_LOG_PATH" | head -n 1)
    if [ -n "$PORT" ] && [ -n "$TOKEN" ]; then
      echo "Jupyter server is running on port: $PORT"
      echo "Jupyter server token: $TOKEN"
      echo "Jupyter can be accessed at: http://localhost:$PORT/?token=$TOKEN"
    else
      echo "Could not find port or token in the log file."
    fi
  else
    echo "Log file not found: $APPTAINER_LOG_PATH"
  fi
}

# Call the function to find Jupyter server port and token
find_jupyter_info

# Remove the log file
rm -f $APPTAINER_LOG_PATH