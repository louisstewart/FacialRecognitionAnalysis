import numpy as np
from hmmlearn.hmm import GaussianHMM
from facerec import functions


def train_models(model, x, y):
    """
    Train a Gaussian HMM for each class in the input matrix.

    For this, a default of 7 states is used. Hence the size of startprob and transmat.

    As this is a Left-Right HMM, the start probability is 100% for state 1, and 0 for all others.

    Args:
        model: HMM class object
        x: input image feature vector
        y: labels for images

    Return:
         Trained Model
    """
    classes = len(np.unique(y))

    if classes < 1:
        raise ValueError("Need at least 1 class to train HMMs with.")

    startprob = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    transmat = np.array([[0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]])

    for image_class in range(classes):
        # Partition the training data into arrays for each class
        # Then train a Hidden Markov Model for each
        covar = "diag" if model.diag else "full"
        hmm = GaussianHMM(model.states, covar, n_iter=20, tol=0.00001, random_state=np.random.RandomState())
        hmm.startprob_ = startprob
        hmm.transmat_ = transmat
        # Loop iterator corresponds to label of data
        hmm.label = image_class
        features = functions.get_labelled_train_data(x, y, image_class)

        # Fit the HMM for that label
        old_block_num = features.shape[1]
        features_size = features.shape[2]
        length = features.shape[0]

        # Need to flatten out the first 2 dimensions of feature vector
        # and pass an array of length into the HMM
        # old_block_num = number of observation blocks for a single image.
        lengths = []
        for i in range(length):
            lengths.append(old_block_num)

        feat = features.reshape((length * old_block_num, features_size))

        hmm.fit(feat, lengths=lengths)  # Train the model for 1 class.

        model.hmms.append(hmm)

    return model


def predict(model, x, y):
    """
    Using a trained model containing HMMs for each class, predict classes
    for a set of images.

    Args:
        model: HMM class
        x: test images
        y: test labels

    Return:
        results vector (list of predicted class labels)
    """
    # We have an HMM for each class, so need to test each one and select the most likely.
    results = np.zeros((len(y),), dtype=np.float32)
    probabilities = np.empty((len(model.hmms, )), dtype=np.float32)

    for (i, sample) in enumerate(x):
        for (image_id, hmm) in enumerate(model.hmms):
            # Test the image against the current hmm
            (res, sts) = hmm.decode(sample, algorithm="viterbi")
            probabilities[image_id] = res
        results[i] = np.argmax(probabilities, axis=0)

    return results
