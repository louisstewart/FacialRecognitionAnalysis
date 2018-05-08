import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
from facerec.reader import ImageType


def get_labelled_train_data(x, y, i):
    """
    Partition the training data into an array that contains only training data for label i

    Args:
        x: the full training set
        y: the label set
        i: the label to extract
    
    Return: 
        training data matching label i
    """
    indices = np.where(y == i)
    return x[indices[0]]


def generate_confusion_matrix(results, y_labels, ids, output, title, normalise=False, display=True):
    """
    Generate and print a confusion matrix for the prediction results.

    Args: 
        results: The predicted labels from the model
        y_labels: The actual labels of the test data
        ids: List of photo IDs
        output: output directory
        title: the title for the generated matrix graph image
        normalise: normalise the data to 0..1 range
        display: show the matrix on screen? Default matplotlib backend here is Agg, so doesn't show to screen.
    Return:
        None
    """
    results = results.astype(int)
    y_labels = y_labels.astype(int)
    length = len(ids)
    matrix = [[0 for x in range(length)] for y in range(length)]
    for x in range(len(results)):
        prediction = results[x]
        actual = y_labels[x]
        matrix[actual][prediction] += 1

    # Show the data on console
    print "\n- - - - Confusion Matrix - - - -\n"
    print "Rows = Actual"
    print "Columns = Predictions\n"
    print "------- Predicted Classes ---------"
    print ' '.join(['{:4}'.format(item) for item in range(1, length + 1)])
    print "___________________________________"
    print "\n------------------------------------\n".join(['|'.join(['{:4}'.format(item)
                                                                     for item in row]) for row in matrix])
    print "------------------------------------\n\n"

    # Show data as a plot, and save plot to output directory.
    mat = np.asarray(matrix, dtype=np.float32)
    labels = [str(i) for i in range(1, len(ids) + 1)]
    my_dpi = 96
    fig = plt.figure(figsize=(2000/my_dpi, 2000/my_dpi), dpi=my_dpi)
    plt.margins(0.05, 0.1)
    ax = fig.add_subplot(111)

    if normalise:
        mat = mat / mat.sum(axis=1, keepdims=True)
    else:
        mat = mat.astype(int)

    im = ax.imshow(mat, interpolation='nearest', cmap=plt.cm.Reds)
    ax.set_title(title)
    ticks = np.arange(length)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    div = make_axes_locatable(ax)
    max = mat.max() * 100 if normalise else mat.max()
    cax = div.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, ticks=[0, max])

    threshold = mat.max() / 2

    for i in range(length):
        for j in range(length):
            # Need to change font colour to make it visible when value is above the threshold
            if mat[i, j] > 0.0:
                ax.text(j, i, ("%.1f" % (mat[i, j] * 100)) if normalise else mat[i, j], fontsize='smaller', ha='center', va='center', color='white' if mat[i, j] > threshold else 'black')

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    plt.tight_layout()
    fig.savefig(output + "/" + title + ".png", dpi=my_dpi*3)
    if display:
        plt.show()


def show_image(matrix, shape, image_type=ImageType.RGB, flattened=False):
    """
    Display an image using the data from a matrix

    If matrix contains multiple image streams then only the first one is used:
    E.g. is matrix is RGB+IR then only RGB is shown.

    Args:
        flattened: is the input array 1D
        shape: The original shape
        matrix: image data
        image_type: 1 of RGB, GRAY, IR, D
    Return:
        None
    """

    rgb = (image_type & ImageType.RGB) > 0
    grey = (image_type & ImageType.GREY) > 0
    ir = (image_type & ImageType.IR) > 0
    d = (image_type & ImageType.D) > 0

    streams = 3 if rgb else 1

    height = shape[0]
    width = shape[1]

    im = np.copy(matrix)
    if flattened:
        if rgb:
            image = im[0:height*width*3].reshape((height, width, 3))
        else:
            image = im[0:height*width].reshape((height, width))
    else:
        image = im.reshape((height, width, streams))[:, :, 0: 3 if rgb else 1]

    plt.axis("off")
    if rgb:
        plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    elif grey or ir:
        plt.imshow(image.reshape(shape), cmap='gray')
    elif d:
        map_depth_to_byte = 8000.0 / 256.0
        max_depth = 4500.0
        min_depth = 900.0
        map_func = np.vectorize(lambda px: (px / map_depth_to_byte) if (min_depth <= px <= max_depth) else 0.0)
        image = map_func(image)
        plt.imshow(image.reshape(shape), cmap='gray')

    plt.show()


def calc_streams(image_stream_type):
    rgb = (image_stream_type & ImageType.RGB) > 0
    grey = (image_stream_type & ImageType.GREY) > 0
    ir = (image_stream_type & ImageType.IR) > 0
    d = (image_stream_type & ImageType.D) > 0

    streams = int(rgb) * 3 + int(grey) + int(ir) + int(d)
    return streams


def get_logger(out_file, model, log_name="log"):
    """
    Generate a logger for the model

    Args:
        out_file: directory to log to
        model: string of model type
        log_name: identifier for the log object
    Return:
        logger
    """
    logger = logging.getLogger('shproject.' + model)
    handler = logging.FileHandler(os.path.join(out_file, log_name+".txt"))
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def handle_filepath(path):
    """
    Expand user, and deal with relative paths.

    Args:
        path: some file string

    Returns:
        handled path
    """
    curr = os.getcwd()
    if path.startswith("~"):
        path = os.path.expanduser(path)
    elif not path.startswith("/"):
        path = os.path.join(curr, path)

    return path
