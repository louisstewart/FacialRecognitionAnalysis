import getopt
import pickle
import sys
import os
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from facerec import functions
from facerec.classifier import SupportVectorMachine, HiddenMarkovModel, ConvolutionalNeuralNetwork
from facerec.feature import PCA, MeanNormalisation, DiscreteCosine, MinMaxNormalisation
from facerec.feature_operators import FeatureChain, BlockExtractor, FeatureMap
from facerec.model import Model, ModelType
from facerec.reader import ImageReader, ImageType
from facerec.validation import F1Score

display = False


def main(argv):
    input_data = ""
    test_data = ""
    output = ""
    cv_data = ""
    model_data = ""
    cross_verify = False
    load = False
    load_model = False
    save = False
    image_stream_type = 0x0
    dist = None
    model_arg = ModelType.SVM
    global display
    arg_string = "usage: %s --input path/to/files --testing-input path/to/files --output output/path " \
                 "--cv-input path/to/cv/data" % argv[0]
    try:
        opts, args = getopt.getopt(argv[1:], "hqi:o:t:v:ls", ["input=", "output=", "testing-input=", "model-data=",
                                                              "cv-input=", "ir", "rgb", "grey", "depth", "1m", "2m",
                                                              "3m", "4m", "svm", "hmm", "cnn"])
    except getopt.GetoptError:
        print arg_string
        sys.exit(2)

    if len(opts) < 1:
        print arg_string
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print "To train"
            print arg_string
            print "OR"
            print "To load and test"
            print "usage: %s -load --input path/to/files --testing-input path/to/files --output output/path" % argv[0]
            sys.exit(0)
        elif opt == "-l":
            load = True
        elif opt == "-s":
            save = True
        elif opt == "-q":
            display = False
        elif opt == "--grey":
            image_stream_type |= ImageType.GREY
        elif opt == "--rgb":
            image_stream_type |= ImageType.RGB
        elif opt == "--ir":
            image_stream_type |= ImageType.IR
        elif opt == "--depth":
            image_stream_type |= ImageType.D
        elif opt == "--1m":
            dist = 1
        elif opt == "--2m":
            dist = 2
        elif opt == "--3m":
            dist = 3
        elif opt == "--4m":
            dist = 4
        elif opt == "--svm":
            model_arg = ModelType.SVM
        elif opt == "--cnn":
            model_arg = ModelType.CNN
        elif opt == "--hmm":
            model_arg = ModelType.HMM
        elif opt == "--model-data":
            load_model = True
            if arg == "":
                print arg_string
                sys.exit(2)
            model_data = arg
        elif opt in ["-i", "--input"]:
            if arg == "":
                print arg_string
                sys.exit(2)
            input_data = functions.handle_filepath(arg)
            if not os.path.exists(input_data):
                print "Error: Input folder does not exist"
                sys.exit(2)
        elif opt in ["--testing-input", "-t"]:
            if arg == "":
                print arg_string
                sys.exit(2)
            test_data = functions.handle_filepath(arg)
            if not os.path.exists(test_data):
                print "Error: Test Input folder does not exist"
                sys.exit(2)
        elif opt in ["--cv-input", "-v"]:
            if arg == "":
                print arg_string
                sys.exit(2)
            cross_verify = True
            cv_data = functions.handle_filepath(arg)
            if not os.path.exists(cv_data):
                print "Error: Cross Verification Input folder does not exist"
                sys.exit(2)
        elif opt in ["-o", "--output"]:
            if arg == "":
                print arg_string
                sys.exit(2)
            output = functions.handle_filepath(arg)
            if not os.path.exists(output):
                os.mkdir(output)

    if image_stream_type == 0:
        image_stream_type |= ImageType.GREY

    # INITIALISE MODEL
    streams = functions.calc_streams(image_stream_type)
    file_string = ImageType.file_friendly_string(image_stream_type)
    dim = (80, 64)  # (80, 64)

    model = create_model(model_arg, output, file_string, dim, streams, image_stream_type, dist, save)

    model.logger.info("Image dimensions: %s", dim)
    im_string = ImageType.stringify(image_stream_type)
    print "Training on images of type: %s" % im_string
    model.logger.info("Training on Image Types: %s", im_string)

    if dist is not None:
        model.logger.info("Using images from distance %d m", dist)

    if load:
        print "Loading saved data from %s ..." % input_data
        ins = os.path.abspath(os.path.join(input_data, "train_data_x.npy"))
        x = np.load(ins)
        ins = os.path.abspath(os.path.join(input_data, "train_data_y.npy"))
        y = np.load(ins)
        ins = os.path.abspath(os.path.join(input_data, "test_data_x.npy"))
        x_test = np.load(ins)
        ins = os.path.abspath(os.path.join(input_data, "test_data_y.npy"))
        y_test = np.load(ins)
        ins = os.path.abspath(os.path.join(input_data, "ids.pkl"))
        with open(ins, 'rb') as fp:
            ids = pickle.load(fp)

        if load_model:
            print "Loading SVM from file."
            if len(model_data):
                ins = os.path.abspath(model_data)
            else:
                ins = os.path.abspath(os.path.join(input_data, "model_data"))
            model.classifier = SupportVectorMachine.load_from_file(ins)

        if cross_verify:
            ins = os.path.abspath(os.path.join(input_data, "cv_data_x.npy"))
            x_cv = np.load(ins)
            ins = os.path.abspath(os.path.join(input_data, "cv_data_y.npy"))
            y_cv = np.load(ins)
            if load_model:
                test(model, ids, x_test, y_test, x_cv, y_cv)
            else:
                train_model(model, ids, x, y, shape=dim)
                test(model, ids, x_test, y_test, x_cv, y_cv)
        else:
            if load_model:
                test(model, ids, x_test, y_test)
            else:
                train_model(model, ids, x, y, shape=dim)
                test(model, ids, x_test, y_test)

    else:
        # Need to read in the data
        in_data = ImageReader(input_data)
        one_hot = model_arg == ModelType.CNN
        [x, y] = in_data.read_images(dim, one_hot=one_hot, image_type=image_stream_type, flatten=model.flattened, distance=dist)
        ids = in_data.get_ids()  # Need to get the correct list of IDs to feed to the other readers

        t_data = ImageReader(test_data, ids=ids)
        [x_test, y_test] = t_data.read_images(dim, one_hot=one_hot, image_type=image_stream_type, flatten=model.flattened, distance=dist)

        if cross_verify:
            cross_data = ImageReader(cv_data, ids=ids)
            [x_cv, y_cv] = cross_data.read_images(dim, one_hot=one_hot, image_type=image_stream_type, flatten=model.flattened, distance=dist)
            if save:
                save_data(model, ids, x, y, x_test, y_test, x_cv, y_cv)  # Save the data before training
            else:
                save_data(model, ids)
            train_model(model, ids, x, y, shape=dim)  # Train the model
            test(model, ids, x_test, y_test, x_cv, y_cv)
        else:
            if save:
                save_data(model, ids, x, y, x_test, y_test)  # Save the data before training
            else:
                save_data(model, ids)
            train_model(model, ids, x, y, shape=dim)
            test(model, ids, x_test, y_test)


def create_model(model_type, output, stream_string, dim, streams, image_stream_type, dist=None, save=False):
    """
    Create a new instance of model, with classifier and feature pre-processing
    Args:
        model_type: enum value of SVM, CNN, HMM
        output: output file location
        stream_string: friendly string of image types
        dim: size of images
        streams: number of image streams
        image_stream_type: image type flags
        dist: distance to filter for (int or None)
        save: save output

    Returns:
        model object composed of classifier and feature extraction
    """
    if model_type == ModelType.SVM:
        classifier = SupportVectorMachine(output, stream_string, dim, dist=dist)
        mu = MeanNormalisation(streams=streams, flattened=classifier.flattened)
        pca = PCA(num_components=150)
        feature = FeatureChain(mu, pca)
    elif model_type == ModelType.HMM:
        l = 16  # Height of block for feature extraction
        p = 12  # Amount of overlap between blocks
        k_size = 25  # DCT kernel size
        diag = True
        classifier = HiddenMarkovModel(output, stream_string, dim, dist=dist, diag=diag)
        feature = FeatureChain(BlockExtractor(dim, streams, classifier.flattened, l=l, p=p),
                               FeatureMap(DiscreteCosine(k_size, (l, dim[1]), streams), k_size, streams))
    else:
        dim = list(dim) + [streams]
        classifier = ConvolutionalNeuralNetwork(output, stream_string, shape=dim, dist=dist)
        feature = MinMaxNormalisation(streams=streams, flattened=False)
    model = Model(feature, classifier, image_stream_type, save_data=save)
    return model


def save_data(model, ids, x=None, y=None, x_test=None, y_test=None, x_cv=None, y_cv=None):
    if x is not None and y is not None:
        np.save("%s/%s" % (model.save_dir, "train_data_x"), x)
        np.save("%s/%s" % (model.save_dir, "train_data_y"), y)
    if x_test is not None and y_test is not None:
        np.save("%s/%s" % (model.save_dir, "test_data_x"), x_test)
        np.save("%s/%s" % (model.save_dir, "test_data_y"), y_test)
    if x_cv is not None and y_cv is not None:
        np.save("%s/%s" % (model.save_dir, "cv_data_x"), x_cv)
        np.save("%s/%s" % (model.save_dir, "cv_data_y"), y_cv)

    print "Saved image data to %s\n" % model.save_dir

    with open("%s/%s" % (model.save_dir, "ids.pkl"), "w") as fp:
        pickle.dump(ids, fp)


# _______________________________________________     Now training      _______________________________________________#
def train_model(model, ids, x, y, shape=(360, 260)):
    """
    Train the specified model, and print some statistics about the training.
    Args:
        model: the model to train
        x: training images
        y: training labels
        ids: list of IDs for images
        shape: image size parameter

    Return:
        trained model
    """
    print "\n---------------------------"
    print "       Training Models       "
    print "---------------------------\n"

    if display:
        print "Look at first & last training images to make sure they're sane."
        functions.show_image(x[0], shape=shape, image_type=model.image_type, flattened=model.flattened)
        idx = y[0].astype(int)
        print "Image1 label = %s : %d" % (ids[idx], idx)
        functions.show_image(x[-2], shape=shape, image_type=model.image_type, flattened=model.flattened)
        idx = y[-2].astype(int)
        print "Image2 label = %s : %d" % (ids[idx], idx)

    print "Starting training"

    start = time.clock()
    model.fit(x, y)
    end = time.clock()

    if model.save_data:
        model.classifier.save("%s/%s" % (model.output, "model_data"))

    model.logger.info("Started training at %f", start)
    model.logger.info("Finished training at %f", end)
    model.logger.info("Train results ------")
    model.logger.info("Classes: %d", len(ids))
    model.logger.info("Time: %f s", (end - start))

    print "Finished training"
    print "\n-------Training Results-------"
    print "Images: %d" % len(y)
    print "Classes: %d" % len(ids)
    print ids
    print "Time: %d s" % (end-start)

    return model


def test(model, ids, x_test, y_test, x_cv=None, y_cv=None):

    print "\nPredicting classes for test set"

    start = time.time()
    results = model.predict(x_test, y_test)  # Predict the labels using the Model we just fit.
    end = time.time()

    model.logger.info("PREDICTION results ------")
    model.logger.info("Started at %f", start)
    model.logger.info("Finished at %f", end)
    model.logger.info("Images: %d", len(y_test))
    model.logger.info("Classes: %d", len(ids))
    model.logger.info("Time: %f s", (end - start))

    print "\n-------Prediction Results For Test Data-------"
    print "Images: %d" % len(y_test)
    print "Classes: %d" % (len(ids))
    print "Time: %f s" % float(end - start)

    if y_test.ndim > 1:  # One-hot encoding, so take argmax
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test

    mask = results == y_test_labels
    correct = np.count_nonzero(mask)
    accuracy = (correct*100.0/results.size)
    print "Accuracy: %f%%" % accuracy

    model.logger.info("Prediction accuracy: %f %%", accuracy)

    scorer = F1Score(model, "micro")
    f1 = scorer.score(results, y_test_labels)

    model.logger.info("F1 Score: %f", f1)
    print "F1 Score: %f" % f1

    functions.generate_confusion_matrix(results, y_test_labels, ids, model.output, "Test Set Confusion Matrix",
                                        normalise=True)

    if x_cv is not None and y_cv is not None:

        start = time.time()
        results = model.predict(x_cv, y_cv)
        end = time.time()

        model.logger.info("CV results ------")
        model.logger.info("Started at %f", start)
        model.logger.info("Finished at %f", end)
        model.logger.info("Images: %d", len(y_cv))
        model.logger.info("Classes: %d", len(ids))
        model.logger.info("Time: %f s", (end - start))

        print "-------Prediction Results For CV Data-------"
        print "Images: %d" % len(y_cv)
        print "Classes: %d" % (len(ids))
        print "Time: %f s" % float(end - start)

        if y_cv.ndim > 1:  # One-hot encoding, so take argmax
            y_cv_labels = np.argmax(y_cv, axis=1)
        else:
            y_cv_labels = y_cv

        mask = results == y_cv_labels
        correct = np.count_nonzero(mask)
        accuracy = (correct*100.0/results.size)

        model.logger.info("CV accuracy: %f %%", accuracy)

        print "Accuracy: %f%%" % accuracy

        functions.generate_confusion_matrix(results, y_cv_labels, ids, model.output, "CV Set Confusion Matrix")


if __name__ == "__main__":
    main(sys.argv)


