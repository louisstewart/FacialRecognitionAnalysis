#!/bin/sh

# Run 5 experiments for each image steam combination,
# saving image numpy arrays after first run for faster
# loading in the subsequent runs.

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --depth

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --depth

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/GREY_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir --depth

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_IR_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --depth

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/RGB_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb --ir --depth

python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir --depth
python py_scripts/facial_recognition_main.py --cnn -l -i ~/Documents/images/IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --ir --depth
