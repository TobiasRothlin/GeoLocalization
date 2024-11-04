#!/bin/bash
SIF_FOLDER="/home/tobias.rothlin/data/SIF"
IMAGE_NAME="$SIF_FOLDER/Tensorboard.sif"
DEFINITION_FILE="/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Tensorboard/Tensorboard.def"
TMUX_SESSION="tensorboard_pipeline"
LOG_FILE="/home/tobias.rothlin/data/Logs/Tensorboard_output_$(date +"%Y-%m-%d_%H-%M-%S").log"
REBUILD=false
RUN_AS_INSTANCE=false
SESSION_NAME="Tensorboard"

APPTAINER_LOG_PATH="/home/tobias.rothlin/.apptainer/instances/logs/sifs-dgx/tobias.rothlin/Tensorboard.err"

# Create the log file
touch $LOG_FILE

# Parse command-line options
while getopts "r" opt; do
  case $opt in
    r)
      REBUILD=true
      ;;
    i)
      RUN_AS_INSTANCE=true
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

# Start the Tensorboard container
if [ "$RUN_AS_INSTANCE" = true ]; then
  echo "Starting the Tensorboard container as an instance..."
  apptainer instance start $IMAGE_NAME $SESSION_NAME
else
  echo "Starting the Tensorboard container..."
  apptainer run $IMAGE_NAME $SESSION_NAME
fi
