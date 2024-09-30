#!/bin/bash

IMAGE_NAME="ImageClassificationPipeline.sif"
DEFINITION_FILE="ImageClassificationPipeline.def"
TMUX_SESSION="image_classification_pipeline_session"
REBUILD=false
CUDA_VISIBLE_DEVICES=""

# Parse command-line options
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

# Build the Apptainer container if rebuild is necessary
if [ "$REBUILD" = true ]; then
  echo "Building the Apptainer container..."
  apptainer build $IMAGE_NAME $DEFINITION_FILE
fi

# Start a new tmux session and run the container
echo "Starting tmux session and running the container..."
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "GPU specified. Running the container on GPU $CUDA_VISIBLE_DEVICES..."
    tmux new-session -d -s $TMUX_SESSION "APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES apptainer run --nv $IMAGE_NAME; sleep 30"
else
    echo "No GPU specified. Running the container on CPU..."
    tmux new-session -d -s $TMUX_SESSION "apptainer run $IMAGE_NAME; sleep 30"
fi

# Check if the tmux session started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start tmux session."
    exit 1
fi

# Attach to the tmux session
tmux attach-session -t $TMUX_SESSION