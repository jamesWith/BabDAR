#!/bin/bash

# Script to run BabDAR all at once
echo -n "Enter path to video file to detect on: "
read videopath
# Run detection

while true; do
    read -p "Would you like to save the detection video [y/n]?" detectionvid
    case $detectionvid in
        [YyNn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

while true; do
    read -p "Would you like to save the tracking video [y/n]?" trackingvid
    case $trackingvid in
        [YyNn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

case $detectionvid in
    [Yy]* ) ./Darknet/darknet detector demo Darknet/data/obj.data \
	Darknet/cfg/yolov4-run.cfg \
	Darknet/Trained-models/yolov4-obj_best.weights  \
	-dont_show $videopath \
	-i 0 -thresh 0.5 -out_filename Detectionout.mp4; break;;
    [Nn]* ) ./Darknet/darknet detector demo Darknet/data/obj.data \
	Darknet/cfg/yolov4-run.cfg \
	Darknet/Trained-models/yolov4-obj_best.weights  \
	-dont_show $videopath \
	-i 0 -thresh 0.5; break;;
    * ) echo "Please answer yes or no.";;
esac

case $trackingvid in
    [Yy]* ) python3 SORT/tracker.py --display True --vidin $videopath; break;;
    [Nn]* ) python3 SORT/tracker.py --display False; break;;
    * ) echo "Please answer yes or no.";;
esac

python3 FaZe/Actiondetect.py baboons \
$videopath \
FaZe/Kinetics_BNInception__rgb_model_best.pth.tar \
FaZe/Kinetics_BNInception__rgbdiff_model_best.pth.tar \
--arch BNInception --classInd_file FaZe/classInd.txt \
-j 1 --num_segments 3 --sampling_freq 6 --delta 1 --score_weights 1 1.5 --quality 256