import numpy as np
import cv2
from abc import ABCMeta, abstractmethod
from sklearn.decomposition import PCA as PrincipalComponentAnalysis


class FeatureExtractor(object):
    """
    Abstract class which represents a feature extractor of some sorts for input data X.

    The one abstract method will be extract(x) which returns a matrix of features.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def extract(self, x): pass


class MeanNormalisation(FeatureExtractor):
    def __init__(self, streams, flattened=False):
        FeatureExtractor.__init__(self)
        self.streams = streams
        self.flattened = flattened

    def extract(self, x):
        for example in x:
            for i in range(self.streams):
                if self.flattened:
                    start = i * len(example) / self.streams
                    end = start + len(example) / self.streams
                    view = example[start:end]
                else:
                    view = example[:, :, i]
                mu = np.mean(view)
                sd = np.std(view)
                view -= mu
                view /= sd

        return x


class MinMaxNormalisation(FeatureExtractor):
    """
    Normalise the data input to within some range.

    Default range is [0.0 .. 1.0]
    """
    def __init__(self, streams, flattened=False, low=0.0, high=1.0,):
        FeatureExtractor.__init__(self)
        self.low = low
        self.high = high
        self.flattened = flattened
        self.streams = streams

    def extract(self, x):
        for example in x:
            for i in range(self.streams):
                if self.flattened:
                    start = i * len(example) / self.streams
                    end = start + len(example)/self.streams
                    view = example[start:end]
                else:
                    view = example[:, :, i]
                x_min = float(np.nanmin(view))
                x_max = float(np.nanmax(view))

                # Normalise X features to range [0.0 .. 1.0]
                view -= x_min
                view /= (x_max - x_min)
                view *= (self.high - self.low)
                view += self.low

        return x


class Resize(FeatureExtractor):
    """
    Resize an input matrix
    """
    def __init__(self, orig_shape=(80, 64), new_shape=(80, 64), streams=1, flattened=True):
        self.orig_shape = orig_shape
        self.new_shape = new_shape
        self.streams = streams
        self.flattened = flattened

    def extract(self, x):
        """
        X may have originally been a 4 dimensional matrix of n_samples x width x height x num_channels
        So, need to know how to reshape X before resizing.
        """
        if self.flattened:
            features = np.empty((len(x), self.new_shape[0] * self.new_shape[1] * self.streams))
        else:
            features = np.empty((len(x), self.new_shape[0], self.new_shape[1], self.streams))  # Gen new matrix.
        for (i, image) in enumerate(x):

            if self.flattened:
                view = None
                for j in range(self.streams):
                    start = j * len(image) / self.streams
                    end = start + len(image)/self.streams
                    im = np.copy(image[start:end]).reshape((self.orig_shape[0], self.orig_shape[1]))
                    im = cv2.resize(im, (self.new_shape[1], self.new_shape[0]))
                    im = im.reshape(-1)
                    if view is None:
                        view = im
                    else:
                        view = np.concatenate((view, im))
            else:
                view = image
                view = cv2.resize(view, (self.new_shape[1], self.new_shape[0]))

            features[i] = view

        return features


class PCA(FeatureExtractor):
    """
    Principal Component analysis on a data-set

    The principal components will be used to reduce the dimension of the input data
    as the top N components will be selected from the data, and an input image will
    then be projected onto these N components.

    """
    def __init__(self, num_components):
        FeatureExtractor.__init__(self)
        self.num_components = num_components
        self.pca = PrincipalComponentAnalysis(n_components=num_components)
        self.fit = False

    def extract(self, x):
        if not self.fit:
            features = self.pca.fit_transform(x)
            self.fit = True
        else:
            features = self.pca.transform(x)
        return features


class DiscreteCosine(FeatureExtractor):
    """
    Discrete cosine transform feature extractor.

    Compute the 2D-DCT of an input matrix, and select a block of the low frequency to return
    """
    def __init__(self, kernel_size, shape, streams, flattened=True):
        self.kernel_size = kernel_size
        self.flattened = flattened
        self.streams = streams
        if len(shape) > 2:
            self.shape = (self.shape[0], self.shape[1])
        else:
            self.shape = shape
        self.indices = self.__precompute_indices(shape)

    def extract(self, x):
        """
        Take the 2D-DCT of the input and return a section of the low frequency as this
        contains the most energy

        as per:

        Kohir, V.V. and Desai, U.B., 1998, October.
        Face recognition using a DCT-HMM approach.
        In Applications of Computer Vision, 1998. WACV'98. Proceedings., Fourth IEEE Workshop on (pp. 226-231). IEEE.

        Args:
            x: input matrix

        Return:
            feature vector of DCT components
        """
        if len(x) < 1:
            raise ValueError("Length of X needs to be larger than 0")

        if self.flattened:
            shape = (x.shape[0], self.kernel_size * self.streams)
        else:
            shape = (x.shape[0], self.kernel_size, self.streams)

        features = np.empty(shape, dtype=np.float32)

        for (idx, image) in enumerate(x):
            feat = None
            # im = image.reshape((self.shape[0], self.shape[1], self.streams))
            for i in range(self.streams):
                if self.flattened:
                    start = i * len(image) / self.streams
                    end = start + len(image) / self.streams
                    view = image[start:end]
                else:
                    view = image[:, :, i]
                res = np.empty(self.shape, dtype=np.float32)
                view = view.reshape(self.shape)
                cv2.dct(view, res)
                f_im = self.__zig_zag(res)

                if feat is None:
                    feat = f_im
                else:
                    if self.flattened:
                        feat = np.concatenate([feat, f_im])
                    else:
                        feat = np.dstack([feat, f_im])

            features[idx] = feat

        return features

    def __zig_zag(self, matrix):
        """
        Zig-Zag scan of a matrix

        as per:

        Kohir, V.V. and Desai, U.B., 1998, October.
        Face recognition using a DCT-HMM approach.
        In Applications of Computer Vision, 1998. WACV'98. Proceedings., Fourth IEEE Workshop on (pp. 226-231). IEEE.

        Args:
            matrix: the DCT components

        Return:
            set of DCT components extracted by zig-zag scanning:

        """
        return matrix.reshape(-1)[self.indices][:self.kernel_size].copy()

    @staticmethod
    def __precompute_indices(shape):
        """
        Pre-compute the indices we will use to extract the DCT co-efficients:
        Adapted from MatLab code given on StackOverflow:
        http://stackoverflow.com/questions/3024939/matrix-zigzag-reordering

        Args:
            shape: dimensions of matrix

        Return:
            A list of indices that will be used to extract DCT components in zig-zag
        """
        [h, w] = shape
        m = np.arange(1, h + 1, dtype=np.float32).reshape((h, 1)) + np.arange(w).T.reshape((1, w))
        m_ = np.power(-1, m)
        m = m + (np.arange(1, h + 1, dtype=np.float32).reshape((h, 1)) / (h + w)) * m_
        idx = np.argsort(m.reshape(-1))
        return idx


class HistogramEqualisation(FeatureExtractor):
    """
    Histogram Equalisation is a method of smoothing out the contrast of an image.
    It can be a useful technique for removing the effects of varying lighting in
    facial recognition.
    """
    def __init__(self, bins=256):
        FeatureExtractor.__init__(self)
        self.bins = bins

    def extract(self, x):
        hists = np.empty(shape=x.shape)
        for (i, image) in enumerate(x):
            hist, bins = np.histogram(image.flatten(), bins=self.bins, normed=True)
            cdf = hist.cumsum()  # cumulative distribution function
            cdf = 255 * cdf / cdf[-1]  # Normalise values to pixel range
            hists[i] = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)  # Linear interpolation of values
        return hists


class NoneExtractor(FeatureExtractor):
    """
    This is a place filler feature extractor,
    so that a model can be created with no feature extraction

    """
    def extract(self, x):
        return x
