from facerec.feature import FeatureExtractor
from facerec.classifier import Classifier
from enum import Enum


class Model(object):
    """
    Representation of a facial recognition model.

    Composed of 2 parts, feature extraction and classifier.
    Feature extraction can an instance of the FeatureExtractor classes,
    or multiple of them chained together
    """
    def __init__(self, feature, classifier, image_type, save_data=False):
        if not isinstance(feature, FeatureExtractor):
            raise TypeError("Feature is not instance of FeatureExtractor")
        if not isinstance(classifier, Classifier):
            raise TypeError("Classifier argument is not instance of Classifier")

        self.feature_extractor = feature
        self.classifier = classifier
        self.image_type = image_type
        self.flattened = classifier.flattened
        self.output = classifier.output
        self.save_dir = classifier.save_dir
        self.logger = classifier.logger
        self.save_data = save_data

    def fit(self, x, y):
        """
        Fit the model parameters using a training image set and
        a set of corresponding labels - Supervised learning.
        :param x: training images
        :param y: training labels
        :return: Time to train model after feature extraction
        """
        features = self.feature_extractor.extract(x)
        self.classifier.train(features, y)

    def predict(self, x, y):
        """
        Predict the class labels for each example in X
        using the classifier defined for this model
        :param x: test images
        :param y: test labels
        :return: list of predicted class labels
        """
        features = self.feature_extractor.extract(x)
        return self.classifier.predict(features, y)

    def get_type(self):
        """
        String version of the classifier type
        :return: string of classifier
        """
        return self.classifier.get_type()

    def save(self, filename):
        """
        Save the model to disk
        :param filename:
        :return:
        """
        self.classifier.save(filename=filename)


class ModelType(Enum):
    CNN = 1
    SVM = 2
    HMM = 3
