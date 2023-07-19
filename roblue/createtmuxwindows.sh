#!/bin/bash

# Set the prefix key for tmux (default is ctrl+b)
tmux_prefix=C-b

# Get list of directories starting with "exp_"
exp_dirs=$(find .. -maxdepth 1 -type d -name 'exp_*' -printf '../%f\n')

echo $exp_dirs

# Loop through each directory and create a new window in tmux
for exp_dir in $exp_dirs; do
  # Create new window with name of directory
  tmux new-window -n "$exp_dir"
  # Change to directory and run command
  tmux send-keys "cd $exp_dir && ./startroblue.sh" C-m
done

# Switch back to original window
tmux select-window -t 0
