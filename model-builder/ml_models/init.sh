#!/bin/bash
set -e
if conda env list | egrep  -w  $conda_env_name
# If conda enviroment already exists, then just run the training code
# otherwise setup the conda environment
then
    source "$HOME/.bashrc";
    python "$ml_training_file"
else
    conda env create -f $conda_file_path;
    # Make RUN commands use the new environment:
    conda init bash && echo 'conda activate "$conda_env_name"' >>  ~/.bashrc;
    source "$HOME/.bashrc";
    python "$ml_training_file"
fi