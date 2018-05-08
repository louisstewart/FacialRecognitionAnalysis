import numpy as np
from facerec.feature import FeatureExtractor


class FeatureChain(FeatureExtractor):
    """
    This class allows feature extractors to be chained together when passing them to a Model

    Both arguments need to be instance of FeatureExtractor
    Can stack multiple feature chains together for multiple feature extraction methods.
    """
    def __init__(self, feature1, feature2):
        if not isinstance(feature1, FeatureExtractor):
            raise TypeError("Feature1 Argument not instance of FeatureExtractor")
        if not isinstance(feature2, FeatureExtractor):
            raise TypeError("Feature2 Argument not instance of FeatureExtractor")
        FeatureExtractor.__init__(self)
        self.first_feature = feature1
        self.second_feature = feature2

    def extract(self, x):
        f = self.first_feature.extract(x)
        return self.second_feature.extract(f)


class BlockExtractor(FeatureExtractor):
    """
    Block extractor is a feature extraction class used for Hidden Markov Models.
    Given a face image of height H and width W, partition the face into blocks
    of height L with a certain amount of overlap P
    """
    def __init__(self, shape, streams, flattened, l=16, p=12):
        """
        :param shape: dimensions of image
        :param streams: number of image channels
        :param flattened: is image a flattened vector
        :param l: height of block
        :param p: width of overlap
        """
        FeatureExtractor.__init__(self)
        overlap = l - p
        h = shape[0]
        w = shape[1]
        if h % 2 != 0:
            raise ValueError("Image height must be multiple of 2")
        if h / overlap % 2 != 0:
            raise ValueError("Image height must be divisible by size of block overlap (l - p")
        self.h = h
        self.w = w
        self.l = l
        self.p = p
        self.streams = streams
        self.flattened = flattened

    def extract(self, x):
        """
        Extract an image into discrete blocks

        This uses the Top-Bottom sampling method suggested by Nefian:
        Nefian, A.V. and Hayes, M.H., 1998, May.
        Hidden Markov models for face recognition. In Acoustics, Speech and Signal Processing, 1998.
        Proceedings of the 1998 IEEE International Conference on (Vol. 5, pp. 2721-2724). IEEE.

        Args:
            x: the input image set

        Return:
            Set of extracted features, where each example in the set contains blocks of pixels extracted
            from the input images.
        """
        overlap = self.l - self.p  # Calculate the overlap difference as the block height - pixel overlap

        blocks = self.h / overlap - self.l / overlap
        features_size = self.l * self.w
        n_samples = len(x)

        if self.flattened:
            shape = (n_samples, blocks, features_size * self.streams)
        else:
            shape = (n_samples, blocks, features_size, self.streams)

        features = np.empty(shape, dtype=np.float32)
        for (idx, image) in enumerate(x):
            for block_index in range(blocks):
                f_block = None
                for i in range(self.streams):
                    if self.flattened:
                        offset = i * len(image) / self.streams
                        block = image[block_index * overlap + offset: block_index * overlap + offset + features_size]
                        if f_block is None:
                            f_block = block
                        else:
                            f_block = np.concatenate([f_block, block])
                    else:
                        block = image[block_index * overlap: block_index * overlap + self.l, :, i]
                        if f_block is None:
                            f_block = block
                        else:
                            f_block = np.dstack([f_block, block])
                features[idx][block_index] = f_block

        return features


class FeatureMap(FeatureExtractor):
    """
    An accessory class for iterating over blocks within training examples and
    applying a feature extractor to those blocks.
    """
    def __init__(self, feature, feature_size, streams):
        if not isinstance(feature, FeatureExtractor):
            raise TypeError("Feature extractor must be instance of FeatureExtractor")
        FeatureExtractor.__init__(self)
        self.feature = feature
        self.feature_size = feature_size
        self.streams = streams

    def extract(self, x):
        if not x.ndim >= 3:
            raise AssertionError("No need to use IterativeOperator unless applying feature matrix of more than 2D")

        shape = [x.shape[0], x.shape[1], self.feature_size * self.streams]

        features = np.empty(shape, dtype=np.float32)  # Create new feature array.

        for (i, example) in enumerate(x):
            feat = self.feature.extract(example)
            features[i] = feat

        return features
