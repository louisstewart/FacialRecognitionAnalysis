#!/bin/sh

python shproject/test/svm_test.py -s -i ~/Documents/train -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey

python shproject/test/svm_test.py -s -i ~/Documents/train -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --depth

python shproject/test/svm_test.py -s -i ~/Documents/train -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir

python shproject/test/svm_test.py -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb

python shproject/test/svm_test.py -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir

python shproject/test/svm_test.py -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --depth

python shproject/test/svm_test.py -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/GREY_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --grey --ir --depth

python shproject/test/svm_test.py -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_IR_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir

python shproject/test/svm_test.py -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --depth

python shproject/test/svm_test.py -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/RGB_IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --rgb --ir --depth

python shproject/test/svm_test.py -s -i ~/Documents/train/ -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir --depth
python shproject/test/svm_test.py -l -i ~/Documents/images/IR_DEPTH_flat -t ~/Documents/test/ -o ~/Documents/svm_out -v ~/Documents/cv/ --ir --depth
