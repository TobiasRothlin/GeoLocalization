#!/bin/bash
SIF_FOLDER="/home/tobias.rothlin/data/SIF"
IMAGE_NAME="$SIF_FOLDER/Visualization.sif"
DEFINITION_FILE="/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Visualization/Visualization.def"
TMUX_SESSION="visualization_pipeline"
LOG_FILE="/home/tobias.rothlin/data/Logs/Visualization_output_$(date +"%Y-%m-%d_%H-%M-%S").log"
REBUILD=false
USE_TMUX=false

# Create the log file
touch $LOG_FILE

# Parse command-line options
while getopts "r" opt; do
  case $opt in
    r)
      REBUILD=true
      ;;
    t)
      USE_TMUX=true
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

if [ "$USE_TMUX" = true ]; then
    echo "Using tmux..."
    tmux new-session -d -s $TMUX_SESSION "script -f -c 'apptainer run $IMAGE_NAME' $LOG_FILE"

    # Check if the tmux session started successfully
    if [ $? -ne 0 ]; then
        echo "Failed to start tmux session."
        exit 1
    fi

    tmux attach-session -t $TMUX_SESSION

else
    echo "Not using tmux..."
    apptainer run $IMAGE_NAME
fi