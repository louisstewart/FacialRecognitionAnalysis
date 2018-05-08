# Facial Recognition Analysis: 
## Investigating the Speed-Accuracy Trade-off

### Analysis of Facial Recognition

This project contains 3 methods of facial recognition: __Convolutional Neural Networks__ using TensorFlow, 
__Hidden Markov Models__, and __Support Vector Machines__.

The code is composed of 4 basic elements: ImageReader, Model, FeatureExtractor, Classifier. Each model needs 
some form of feature extraction (defined by the children of FeatureExtractor base class), and a classifier.

The model is fed images read in using the following method.
```python
ImageReader.read_images(shape=(80,64), one_hot=False,
                        image_type=ImageType.RGB, flattened=False)
```
This allows for multiple image streams to be specified for training and testing.

The methods of feature extraction used were the result of the Literature Review conducted (contained in the paper). 
For Support Vector Machines this is _Principal Component Analysis_, HMM is _Discrete Cosine Transform_, 
and CNN just uses _Mean Normalisation_ as input pre-processing - all other feature extraction filters are learned by the
network during the end-to-end training process.

### Setup

First, create a virtual environment and install packages.
```bash
virtualenv -p python2.7 venv
source venv/bin/activate
pip install -r requirements.txt
```
or if GPU neural net training is desired, use `requirements_gpu.txt`

Then, set the PYTHONPATH env variable:
```bash
export PYTHONPATH=$PYTHONPATH:~/path/to/project
```

If using tensorflow GPU, don't forget to set LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
```

### Data

The data used can be found at [St Andrews Faces Archive](https://archive.cs.st-andrews.ac.uk/faces/).

Data must then be split into Train, Test and Cross-Validation data-sets.
To do this, the `train_test_split.py` file in `py_scripts` can be used. 
It takes as argument the directory where the data sits.

Example: assuming data is in folder called Manifold in previous directory
level

```bash
python py_scripts/train_test_split.py ../Manifold/
``` 

This script splits each individual's image folder into a train, test and
cv folders **per person** (no images from train appear in either 
test or cv and vice-versa). To combine these into single **train**, 
**test** and **cv** folders use:

``bash
python py_scripts/tt_move.py
``

This script takes no arguments, but assumes that the current directory is
the one which contains each of the individual train, test and cv folders.

Alternatively use the following bash script to do it all in one go:

```bash
bash_scripts/split.sh ../path/to/image/folder
```


### Testing

The main test script for running all of these classifiers is
contained in `py_scritps/facial_recognition_analysis.py`.

It takes a number of arguments:

`--svm`, `--cnn` and `--hmm` are used to specify model type (with
above feature extraction).

`-s` flag is used to save images to file serialised numpy arrays 
after loading (useful for faster run next time)

`-l` flag is used to specify that images should be loaded from 
serialised numpy arrays, instead of raw images. 
If `-l` is given, then directories for `-i`,`-t`,`-v` must 
contain numpy arrays in serialised format.

`-i`, `-t` and `-v` are used with a directory argument following to specify
the directory were the **i**nput, **t**est and **v**alidation data are

`-o` with a directory name following specifies the output directory

`--grey`, `--rgb`, `--ir`, and `--depth` can be used to specify which
image streams to use. Can use multiple at once, except `--rgb` and `--grey` 
which cannot be used with each other.

`--1m`, `--2m`, `--3m` and `--4m` can be used to filter images to one 
distance only. E.g. 1 metre images with `--1m`.

Full example, load images from directories called `train`, `test` and `cv` 
in the Documents folder. Use CNN with Grey + IR images from 1 metre only:

```bash
python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --grey --ir --1m
```

Full example, load images from directories called `train`, `test` and `cv` 
in the Documents folder. Use SVM with all RGB images:

```bash
python py_scripts/facial_recognition_main.py --cnn -s -i ~/Documents/train -t ~/Documents/test/ -o ~/Documents/cnn_out -v ~/Documents/cv/ --rgb
```
