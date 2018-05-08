# Facial Recognition Analysis: Investigating the Speed-Accuracy Trade-off

## Analysis of Facial Recognition

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

## Testing

There are test scripts for running all of these classifiers contained in `test/`.

The data used can be found at [insert link]().



First, create a virtual environment and install packages.
```bash
virtualenv -p python2.7 venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, set the PYTHONPATH env variable:
```bash
export PYTHONPATH=$PYTHONPATH:~/path/to/project
```

Then run one of the testing scripts:
```bash
python project/test/svm_test.py -i ~/path/to/1m_train \
-t ~/path/to/1m_test -o ~/output_path --grey
```

Each test script allows for specifying the image streams used. Options are Greyscale, IR, RGB, Depth in any combination (except those with RGB and Greyscale).
E.g.
"--grey --ir", "--ir --depth", "--rgb --ir --depth", "--rgb"

The 4 test scripts are "svm_test.py", "cnn_test.py", "hmm_test.py" and "cv_test.py". The first 3 test each classifier type, and the last one shows different methods of Cross Verification (either Leave One Out, K Fold or a custom verification using user defined test sets). Options for "cv_test.py" are "--looc", "--kfold" or nothing for custom verification.
e.g.

```bash
python project/test/cv_test.py -i ~/path/to/1m_train \
-t ~/path/to/1m_test -o ~/output_path --grey --ir --kfold
```
