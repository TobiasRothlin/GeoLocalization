#!/bin/bash

SIF_FOLDER="/home/tobias.rothlin/data/SIF"
IMAGE_NAME="$SIF_FOLDER/GlobalClassificationTraining.sif"
DEFINITION_FILE="/home/tobias.rothlin/GeoLocalization/src/DGX1/src/GlobalClassificationTraining/GlobalClassificationTraining.def"
TMUX_SESSION="classification_training_pipeline_session"
LOG_FILE="/home/tobias.rothlin/data/Logs/Training_output_$(date +"%Y-%m-%d_%H-%M-%S").log"


REBUILD=false
CUDA_VISIBLE_DEVICES=""
USE_TMUX=false

# Create the log file
touch $LOG_FILE

# Parse command-line options
while getopts "rg:" opt; do
  case $opt in
    r)
      REBUILD=true
      ;;
    g)
      CUDA_VISIBLE_DEVICES=$OPTARG
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
fi

# Start a new tmux session and run the container
echo "Starting session and running the container..."
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "GPU specified. Running the container on GPU $CUDA_VISIBLE_DEVICES..."

    if [ "$USE_TMUX" = true ]; then
        echo "Using tmux..."
        tmux new-session -d -s $TMUX_SESSION "script -f -c 'APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES apptainer run --nv $IMAGE_NAME' $LOG_FILE"
    else
        echo "Not using tmux..."
        APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES apptainer run --nv $IMAGE_NAME
    fi

   
else
    echo "No GPU specified. Running the container on CPU..."

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
fi

echo "Training Done."