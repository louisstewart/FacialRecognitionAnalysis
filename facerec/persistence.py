from sklearn.externals import joblib


def save(filename, obj):
    """
    Persist an object to disk
    Args:
        filename: location to put file
        obj: object to save
    Return:
        None
    """
    joblib.dump(obj, filename + ".pkl")


def load(filename):
    """
    Load an object from a file
    Args:
        filename: location of data
    Return:
        New object from file
    """
    return joblib.load(filename)
