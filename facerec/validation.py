import time
import numpy as np
from abc import abstractmethod
from facerec import functions


class Scorer(object):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def score(self, y_pred, y_actual): pass


class F1Score(Scorer):
    def __init__(self, model, average="micro"):
        Scorer.__init__(self, model)
        self.average = average

    def score(self, y_pred, y_actual):
        return f1_score(y_pred, y_actual, self.average)


def f1_score(y_pred, y_actual, average):
    return f_beta_score(y_pred, y_actual, 1, average)


def f_beta_score(y_pred, y_actual, beta, average):
    _, _, fb = precision_recall_fscore(y_pred, y_actual, beta, average=average)
    return fb


def precision_recall_fscore(y_pred, y_actual, beta, average="micro"):
    """
    Compute precision, recall and F_beta score of the predicted labels

    Precision = tp / (tp+fp) where tp is the number of true positives (correctly labelled prediction)
    and fp is the false positives.

    Recall is tp (tp + fn), where fn is the number of false negatives (incorrectly labelled as another class)

    F_beta is a weighted mean of precision and recall, with the F1 score being F_beta with beta set to 1,
    so precision and recall are equallty weighted in the output

    As this is multi-class classification there are 2 options for precision, recall and f_score.
    Either the micro average, or the macro average. Macro averaging we calculate the precisions for each individual
    class in the model. For micro we calculate for the whole model.

    f_beta = (1 + beta^2) . (precision . recall) / ((beta^2 . precision) + recall)

    Args:
        y_pred: predicted labels
        y_actual: the actual class labels
        beta: weighting factor - set to 1 for equally weighted F1 score
        average: Micro or Macro precision calculations

    Returns:
        precision: the precision score (tp / tp+fp)
        recall: recall score (tp / tp+fn)
        f_score: weighted mean of precision and recall
    """
    labels = np.unique(y_actual)
    lab_length = len(labels)
    tp = y_pred == y_actual
    tp_b = y_actual[tp].astype(np.int64)  # True positive count
    y_actual = y_actual.astype(np.int64)  # Actual class labels
    y_pred = y_pred.astype(np.int64)      # Predictions

    if len(tp_b):
        tp_sum = np.bincount(tp_b, None, minlength=lab_length).astype(np.float32)
    else:
        tp_sum = np.array([0])
    if len(y_actual):
        actual_sum = np.bincount(y_actual, None, minlength=lab_length).astype(np.float32)
    else:
        raise AssertionError("Must have true class labels to compute precision, recall and F")
    if len(y_pred):
        pred_sum = np.bincount(y_pred, None, minlength=lab_length).astype(np.float32)
    else:
        raise AssertionError("Must have predicted labels to compute precision, recall and F")

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        actual_sum = np.array([actual_sum.sum()])

    beta **= 2  # Square beta for score calculations.
    precision = tp_sum / pred_sum
    recall = tp_sum / actual_sum
    f_score = (1 + beta) * (precision * recall / ((beta * precision) + recall))
    f_score[tp_sum == 0] = 0.0  # Deal with any cases where there were no true positives for a class.

    return precision, recall, f_score


class Validation(object):
    """
    Class interface representing a validation of a model.

    Validation is used to assess how the results of the model
    generalise to an independent data set
    """

    def __init__(self, model, logger, scorer=None):
        self.model = model
        self.logger = logger
        self.scorer = scorer

    @abstractmethod
    def validate(self, xs, ys): pass

    @staticmethod
    def accuracy(true_pos, true_neg, false_pos, false_neg):
        tp = float(true_pos)
        tn = float(true_neg)
        fp = float(false_pos)
        fn = float(false_neg)

        return (tp + tn)/(tp+fp+tn+fn)


class KFoldCrossValidation(Validation):
    """
    Partition the input data into k equally sized sub samples.
    1 of these is held back and used for training, while the
    remaining k-1 samples are used for training.

    Each class is in each of the k folds. But could equally do k-folds without
    paying attention to the classes.
    """
    def __init__(self, model, folds, scorer=None):
        logger = functions.get_logger(model.output, model.get_type() + ".KFOLD", log_name="validation")
        Validation.__init__(self, model, logger, scorer)
        self.folds = folds

    def validate(self, xs, ys):
        classes = len(np.unique(ys))

        per_class_folds = []
        for i in range(classes):
            # Create k data partitions of each class
            class_indices = np.where(ys == i)[0]  # Indices in xs/ys which are an image of this class
            class_folds = np.array_split(class_indices, self.folds)
            per_class_folds.append(class_folds)  # Add the k folds to the list for each class

        self.logger.info("Beginning K-Fold Validation for %d folds", self.folds)
        print "Beginning K-Fold Validation for %d folds" % self.folds

        for f in range(self.folds):
            self.logger.info("\nFold %d", f+1)
            print "\nFold %d" % (f+1)

            fold_indices = per_class_folds[0][f]
            for c in range(1, classes):
                fold_indices = np.concatenate([fold_indices, per_class_folds[c][f]])

            rest = [i for i in range(len(ys)) if i not in fold_indices]

            x_train = xs[rest]
            y_train = ys[rest]
            x_test = xs[fold_indices]
            y_test = ys[fold_indices]

            self.__train(x_train, y_train)

            self.__test(x_test, y_test)

    def __train(self, xs, ys):
        start = time.clock()
        self.model.fit(xs, ys)
        end = time.clock()

        self.logger.info("Train time: %f s", (end - start))
        print "Train time: %f s" % (end - start)

    def __test(self, x_test, y_test):
        start = time.clock()
        results = self.model.predict(x_test, y_test)
        end = time.clock()

        self.logger.info("Predict time: %f s", (end - start))
        print "Predict time: %f s" % (end - start)

        if y_test.ndim > 1:  # One-hot encoding, so take argmax
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test

        mask = results == y_test_labels
        acc = np.count_nonzero(mask) * 100.0 / results.size

        self.logger.info("Overall Test accuracy = %f %%", acc)
        print "Overall Test accuracy = %f %%" % acc

        if self.scorer is not None:
            f1 = self.scorer.score(results, y_actual=y_test_labels)
            self.logger.info("F1 Score = %f %%", f1)
            print "F1 Score = %f %%" % f1
        else:
            classes = np.unique(y_test_labels)
            for c in classes:
                true_positives, false_positives, true_negatives, false_negatives = (0, 0, 0, 0)
                for i in range(len(results)):
                    if results[i] == c and y_test_labels[i] == c:
                        true_positives += 1
                    elif y_test_labels[i] == c:
                        false_negatives += 1
                    elif results[i] == c:
                        false_positives += 1
                    else:
                        true_negatives += 1

                if true_positives > 0:
                    precision = Validation.accuracy(true_positives, 0, false_positives, 0)
                    recall = Validation.accuracy(true_positives, 0, 0, false_negatives)
                else:
                    precision = 0
                    recall = 0

                self.logger.info("Class %d results:", c)
                self.logger.info("precision: %f", precision)
                self.logger.info("recall: %f", recall)
                self.logger.info("---------------------")

                print "Class %d results:" % c
                print "precision: %f" % precision
                print "recall: %f" % recall
                print "---------------------"


class LeaveOneOutValidation(Validation):
    """
    Leave one out validation uses the entire image set minus 1 image as the training set
    The 1 left out is then used to validate.

    This is repeated N times for each input image.
    """
    def __init__(self, model, scorer=None):
        logger = functions.get_logger(model.output, model.get_type() + ".LOOCV", log_name="validation")
        Validation.__init__(self, model, logger, scorer)

    def validate(self, xs, ys):

        self.logger.info("Beginning LOOCV Validation")
        print "Beginning LOOCV Validation"

        for f in range(len(ys)):
            self.logger.info("\nImage %d", f + 1)
            print "\nImage %d" % (f + 1)

            rest = [i for i in range(len(ys)) if i != f]

            x_train = xs[rest]
            y_train = ys[rest]
            x_test = np.array([xs[f]])
            y_test = np.array([ys[f]])

            self.__train(x_train, y_train)

            self.__test(x_test, y_test)

    def __train(self, xs, ys):
        start = time.clock()
        self.model.fit(xs, ys)
        end = time.clock()

        self.logger.info("Train time: %f s", (end - start))
        print "Train time: %f s" % (end - start)

    def __test(self, x_test, y_test):
        start = time.clock()
        results = self.model.predict(x_test, y_test)
        end = time.clock()

        self.logger.info("Predict time: %f s", (end - start))
        print "Predict time: %f s" % (end - start)

        if y_test.ndim > 1:  # One-hot encoding, so take argmax
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test

        mask = results == y_test_labels
        acc = np.count_nonzero(mask) * 100.0 / results.size

        self.logger.info("Overall Test accuracy = %f %%", acc)
        print "Overall Test accuracy = %f %%" % acc

        if self.scorer is not None:
            f1 = self.scorer.score(results, y_actual=y_test_labels)
            self.logger.info("F1 Score = %f %%", f1)
            print "F1 Score = %f %%" % f1
        else:
            classes = np.unique(y_test_labels)
            for c in classes:
                true_positives, false_positives, true_negatives, false_negatives = (0, 0, 0, 0)
                for i in range(len(results)):
                    if results[i] == c and y_test_labels[i] == c:
                        true_positives += 1
                    elif y_test_labels[i] == c:
                        false_negatives += 1
                    elif results[i] == c:
                        false_positives += 1
                    else:
                        true_negatives += 1

                if true_positives > 0:
                    precision = Validation.accuracy(true_positives, 0, false_positives, 0)
                    recall = Validation.accuracy(true_positives, 0, 0, false_negatives)
                else:
                    precision = 0
                    recall = 0

                self.logger.info("Class %d results:", c)
                self.logger.info("precision: %f", precision)
                self.logger.info("recall: %f", recall)
                self.logger.info("---------------------")

                print "Class %d results:" % c
                print "precision: %f" % precision
                print "recall: %f" % recall
                print "---------------------"


class CustomValidation(Validation):
    def __init__(self, model, x_test, y_test, x_cv=None, y_cv=None, scorer=None):
        logger = functions.get_logger(model.output, model.get_type() + ".custom", log_name="validation")
        Validation.__init__(self, model, logger, scorer)
        self.x_test = x_test
        self.y_test = y_test
        self.x_cv = x_cv
        self.y_cv = y_cv
        self.fit = False

    def validate(self, xs, ys):
        self.logger.info("Beginning Custom Validation")
        print "Beginning Custom Validation"

        if self.x_test is not None and self.y_test is not None:
            self.__train(xs, ys)
            self.fit = True
            self.__test(self.x_test, self.y_test)
        if self.x_cv is not None and self.y_cv is not None:
            if not self.fit:
                self.__train(xs, ys)
            self.__test(self.x_cv, self.y_cv)

    def __train(self, xs, ys):
        start = time.clock()
        self.model.fit(xs, ys)
        end = time.clock()

        self.logger.info("Train time: %f s", (end - start))
        print "Train time: %f s" % (end - start)

    def __test(self, x_test, y_test):
        start = time.clock()
        results = self.model.predict(x_test, y_test)
        end = time.clock()

        self.logger.info("Predict time: %f s", (end - start))
        print "Predict time: %f s" % (end - start)

        if y_test.ndim > 1:  # One-hot encoding, so take argmax
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test

        mask = results == y_test_labels
        acc = np.count_nonzero(mask) * 100.0 / results.size

        self.logger.info("Overall Test accuracy = %f %%", acc)
        print "Overall Test accuracy = %f %%" % acc

        if self.scorer is not None:
            f1 = self.scorer.score(results, y_actual=y_test_labels)
            self.logger.info("F1 Score = %f %%", f1)
            print "F1 Score = %f %%" % f1
        else:
            classes = np.unique(y_test_labels)
            for c in classes:
                true_positives, false_positives, true_negatives, false_negatives = (0, 0, 0, 0)
                for i in range(len(results)):
                    if results[i] == c and y_test_labels[i] == c:
                        true_positives += 1
                    elif y_test_labels[i] == c:
                        false_negatives += 1
                    elif results[i] == c:
                        false_positives += 1
                    else:
                        true_negatives += 1
                if true_positives > 0:
                    precision = Validation.accuracy(true_positives, 0, false_positives, 0)
                    recall = Validation.accuracy(true_positives, 0, 0, false_negatives)
                else:
                    precision = 0
                    recall = 0

                self.logger.info("Class %d results:", c)
                self.logger.info("precision: %f", precision)
                self.logger.info("recall: %f", recall)
                self.logger.info("---------------------")

                print "Class %d results:" % c
                print "precision: %f" % precision
                print "recall: %f" % recall
                print "---------------------"
