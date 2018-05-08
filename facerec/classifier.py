import datetime

import numpy as np
import os
import shutil
from abc import ABCMeta, abstractmethod
from sklearn import svm as sksvm
from sklearn.externals import joblib
import facerec.convolution as convolution
from facerec import markov
from facerec import persistence, functions


class Classifier(object):
    __metaclass__ = ABCMeta

    def __init__(self, output, file_string, dim, dist=None, flattened=True):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Create a save directory for the Model data.
        output = os.path.join(output, "%s_%s_%s%s_%s" % (self.get_type(), now_str, file_string, dim[0], dim[1]))
        if not os.path.exists(output):
            os.makedirs(output)

        self.flattened = flattened
        self.__output = output

        sd = os.path.expanduser("~/Documents/images/")  # Default save locations
        sd = os.path.join(sd, file_string)
        sd += "%s_" % dist if dist is not None else ""

        self.save_dir = sd
        self.__logger = functions.get_logger(self.output, self.get_type())

    @abstractmethod
    def train(self, x, y): pass

    @abstractmethod
    def predict(self, x, y): pass

    @abstractmethod
    def get_type(self): pass

    @property
    def logger(self):
        return self.__logger

    @property
    def output(self):
        return self.__output

    @abstractmethod
    def save(self, filename): pass


class SupportVectorMachine(Classifier):
    """

    SVM is a wrapper class for the SciKit Learn SVM module,
    which is itself a wrapper for the libSVM. Here we use a Linear Kernel classifier,
    as this has been shown to be favourable for cases where there are significantly more
    features than classes. (see https://www.coursera.org/learn/machine-learning).

    """
    def __init__(self, output, file_string, shape, dist=None, c=2.67, gamma=5.383):
        Classifier.__init__(self, output, file_string, shape, dist=dist, flattened=True)
        self.save_dir += "flat"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.svm = sksvm.LinearSVC(C=c, random_state=np.random.RandomState())
        self.C = c
        self.gamma = gamma

    def train(self, x, y):
        self.svm.fit(x, y)

    def predict(self, x, y):
        results = self.svm.predict(x)
        return results

    def update(self, svm):
        self.svm = svm

    @staticmethod
    def load_from_file(filename, file_string, output, shape):
        """
        Factory method to load and create an SVM model from persisted data

        Args:
            filename: Path to save model to
            file_string: string of file types used (image channels)
            output: directory to output model to
            shape: image size (for directory naming)

        Returns:
            sklearn LinearSVC created from saved model
        """
        svm_data = joblib.load(filename + ".pkl")
        svm = SupportVectorMachine(output, file_string, shape)
        svm.update(svm_data)
        return svm

    def save(self, filename):
        """
        Persist model to disc

        Args:
            filename: path to save to

        Returns:
            None
        """
        if not os.path.exists(filename):
            os.mkdir(filename)
        outfile = os.path.join(filename, self.get_type())
        persistence.save(outfile, self.svm)

    def get_type(self):
        """
        Return a string specifying what type of classifier this is.

        Returns:
             string name
        """
        return "SVM"


class HiddenMarkovModel(Classifier):
    """
    N state Hidden Markov Model classifier

    Wrapper around HMM Learn Gaussian HMM class.
    """
    def __init__(self, output, file_string, shape, dist=None, diag=True, states=7):
        Classifier.__init__(self, output, file_string, shape, dist=dist, flattened=True)
        self.save_dir += "flat"  # Uses flattened images (1D)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.hmms = []
        self.states = states
        self.shape = shape
        self.h = shape[0]
        self.w = shape[1]
        self.diag = diag

    def train(self, x, y):
        markov.train_models(self, x, y)

    def predict(self, x, y):
        return markov.predict(self, x, y)

    def save(self, filename):
        """
        Persist each trained hmm in the object to disc in a separate file.

        Args:
            filename: save file location

        Returns:
             None
        """
        for (i, hmm) in enumerate(self.hmms):
            if not os.path.exists(filename):
                os.mkdir(filename)
            outfile = os.path.join(filename, "%s_%d" % (self.get_type(), i))
            persistence.save(outfile, hmm)

    @staticmethod
    def load_from_file(filename, file_string, output, shape):
        """
        Load in hmm from save file location.

        To rebuild HMM, need to specify the last 3 params
        in order to give output file a useful name
        Args:
            filename: save file location
            file_string: stringified names of input image stream types
            output: output file location
            shape: size of images (px)

        Returns:
            loaded model
        """
        os.chdir(filename)
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        hmm = HiddenMarkovModel(output, file_string, shape)
        for f in files:
            tokens = f.split("_")
            idx = int(tokens[-1].split(".")[0])
            f_in = os.path.join(filename, f)
            mdl = persistence.load(f_in)
            hmm.hmms.insert(idx, mdl)
        return hmm

    def get_type(self):
        return "HMM"


class ConvolutionalNeuralNetwork(Classifier):
    def __init__(self, output, file_string, shape, dist=None):
        Classifier.__init__(self, output, file_string, shape, dist=dist, flattened=False)
        self.save_dir += "full"  # Uses full 2D images, not flattened
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.shape = shape
        self.out_dir = os.path.join(self.output, "CNN_data")

    def train(self, x, y):
        convolution.conv_net(x, y, size=self.shape, train=True, save_dir=self.out_dir)

    def predict(self, x, y):
        results = convolution.conv_net(x, y, size=self.shape, train=False, save_dir=self.out_dir)
        return results

    def save(self, filename):
        pass   # Model checkpoints already saved during training.

    def cleanup(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def get_type(self):
        return "CNN"
