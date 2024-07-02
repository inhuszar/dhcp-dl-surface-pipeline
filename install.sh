#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if "mamba" is available
if command_exists mamba; then
    EXECUTABLE="mamba"
elif command_exists conda; then
    EXECUTABLE="conda"
else
    echo "Neither 'mamba' nor 'conda' executable found in PATH."
    exit 1
fi

# create new conda environment
$EXECUTABLE create -n dhcp -y
$EXECUTABLE activate dhcp

# install pytorch
$EXECUTABLE install pytorch=1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y

# install dependencies
pip install tqdm scipy==1.10.1 nibabel==5.0.1 antspyx==0.3.8
