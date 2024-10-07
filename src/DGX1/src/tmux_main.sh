echo "Starting main.sh with tmux"

tmux new-session -d -s main "script -f -c 'bash /home/tobias.rothlin/GeoLocalization/src/DGX1/src/main.sh'"