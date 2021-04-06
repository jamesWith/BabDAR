#!/bin/bash




conda env create -f BabDAR-venv.yml
conda init zsh
source ~/.zshrc
conda activate BabDARvenv

echo $CONDA_PREFIX

#Move into the Darknet folder to make darknet
cd Darknet
while true; do
    read -p "Is a GPU available on this computer?" yn
    case $yn in
        [Yy]* ) sed -i '' -e 's/OPENCV=0/OPENCV=1/' Makefile # change makefile to have GPU and OPENCV enabled (only if GPU is available)
				sed -i '' -e 's/GPU=0/GPU=1/' Makefile
				sed -i '' -e 's/CUDNN=0/CUDNN=1/' Makefile
				sed -i '' -e 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile; break;;
        [Nn]* ) sed -i '' -e 's/OPENCV=1/OPENCV=0/' Makefile # change makefile to have GPU and OPENCV enabled (only if GPU is available)
                sed -i '' -e 's/GPU=1/GPU=0/' Makefile
                sed -i '' -e 's/CUDNN=1/CUDNN=0/' Makefile
                sed -i '' -e 's/CUDNN_HALF=1/CUDNN_HALF=0/' Makefile; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# make darknet
make

cd ..
