#!/bin/bash

# Define variables
IMAGE_NAME="test.sif"
DEFINITION_FILE="test.def"
TMUX_SESSION="apptainer_session"

# Build the Apptainer container
echo "Building the Apptainer container..."
apptainer build $IMAGE_NAME $DEFINITION_FILE

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Failed to build the Apptainer container."
    exit 1
fi

# Start a new tmux session and run the container
echo "Starting tmux session and running the container..."
tmux new-session -d -s $TMUX_SESSION "apptainer run $IMAGE_NAME"

# Check if the tmux session started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start tmux session."
    exit 1
fi

echo "Apptainer container is running in tmux session: $TMUX_SESSION"