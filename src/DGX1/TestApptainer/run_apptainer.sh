#!/bin/bash

IMAGE_NAME="test.sif"
DEFINITION_FILE="test.def"
TMUX_SESSION="apptainer_session"
LOG_FILE="/home/tobias.rothlin/data/Logs/apptainer_output_$(date +"%Y-%m-%d_%H-%M-%S").log"

# Create the log file
touch $LOG_FILE



# Build the Apptainer container
echo "Building the Apptainer container..."
apptainer build $IMAGE_NAME $DEFINITION_FILE

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Failed to build the Apptainer container."
    exit 1
fi

# Start a new tmux session and run the container, redirecting output to a log file and console
echo "Starting tmux session and running the container..."
tmux new-session -d -s $TMUX_SESSION "script -f -c 'apptainer run $IMAGE_NAME' $LOG_FILE"

# Check if the tmux session started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start tmux session."
    exit 1
fi

echo "Apptainer container is running in tmux session: $TMUX_SESSION"
echo "Output is being logged to: $LOG_FILE"

# Attach to the tmux session to view the output
tmux attach-session -t $TMUX_SESSION