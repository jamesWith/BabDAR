#!/bin/bash

#Move into the Darknet folder to make darknet
cd Darknet

while true; do
    read -p "Is a GPU available on this computer?" yn
    case $yn in
        [Yy]* ) sed -i 's/OPENCV=0/OPENCV=1/' Makefile # change makefile to have GPU and OPENCV enabled (only if GPU is available)
				sed -i 's/GPU=0/GPU=1/' Makefile
				sed -i 's/CUDNN=0/CUDNN=1/' Makefile
				sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# make darknet
make

cd ..

pip install filterpy==1.4.5
pip install scikit-image==0.17.2
pip install lap==0.4.0
pip install -q xlrd
pip install torchcam