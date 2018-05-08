import cv2
import numpy as np
import os
import glob


class ImageType(object):
    """
    Enum class defining the type of images used for training.
    """
    RGB = 0x1
    GREY = 0x2
    IR = 0x4
    D = 0x8

    @staticmethod
    def stringify(image_type_flag):
        out_str = ""
        if image_type_flag > 15:
            raise AssertionError("Trying to stringify illegal image flag")

        if image_type_flag & ImageType.RGB > 0:
            out_str += "ImageType.RGB "
        if image_type_flag & ImageType.GREY > 0:
            out_str += "ImageType.GREY "
        if image_type_flag & ImageType.IR > 0:
            out_str += "ImageType.IR "
        if image_type_flag & ImageType.D > 0:
            out_str += "ImageType.DEPTH "

        return out_str

    @staticmethod
    def file_friendly_string(image_type_flag):
        out_str = ""
        if image_type_flag > 15:
            raise AssertionError("Trying to stringify illegal image flag")

        if image_type_flag & ImageType.RGB > 0:
            out_str += "RGB_"
        if image_type_flag & ImageType.GREY > 0:
            out_str += "GREY_"
        if image_type_flag & ImageType.IR > 0:
            out_str += "IR_"
        if image_type_flag & ImageType.D > 0:
            out_str += "DEPTH_"

        return out_str


class ImageReader(object):
    """
    Image reading class

    Reads images from an input directory and returns numpy array containing
    the data to be processed.

    """
    def __init__(self, in_file, ids=None):
        self.in_file = in_file
        self.ids = ids
        if ids is None:
            self.ids = []

    def get_ids(self):
        return self.ids

    def num_classes(self):
        return len(self.ids)

    def read_images(self, shape=(80, 64), one_hot=False, image_type=ImageType.RGB, flatten=False, distance=None):
        """
        Read in images from directory. Return numpy matrix of images X and numpy array of labels Y
        Can optionally make y return as a One Hot encoded array of vectors.

        If there is more than one channel for the image, then the image is read in as a matrix (height, width, channels)
        Otherwise, the image is flattened into a 1D vector.

        Args:
            shape: Image dimensions to resize input to
            one_hot: Output label array as array of one_hot vectors (for TensorFlow)
            image_type: Bit flag specifying what image types are being read in.
            flatten: Flag specifying whether to flatten images to 1D arrays
            distance: int - filter images by distance in filename

        Returns:
            List [X,y,ids]
                X: numpy matrix of images
                y: numpy array of labels
                ids: list of unique ids
        """
        if len(self.in_file) < 2:
            raise ValueError("No input file specified")

        # Look at the type parameter to discern what data streams the user wants.
        rgb = (image_type & ImageType.RGB) > 0
        grey = (image_type & ImageType.GREY) > 0
        colour = (rgb or grey)
        ir = (image_type & ImageType.IR) > 0
        d = (image_type & ImageType.D) > 0

        num_channels = int(rgb)*3 + int(grey) + int(ir) + int(d)

        if rgb and grey:
            raise AssertionError("Cannot use RGB and Greyscale simultaneously")

        print "Reading data from: %s" % self.in_file

        if not len(self.ids):
            self.ids = []

        # Because the directories are read in sorted, index of colour image is same as corresponding IR and Depth
        os.chdir(self.in_file)  # Change to the correct directory
        colour_images = sorted(glob.glob("*.tiff"))
        ir_images = sorted(glob.glob("*ir.csv"))
        depth_images = sorted(glob.glob("*depth.csv"))

        height = shape[0]
        width = shape[1]

        file_count = 0
        indices = []

        # To form a one hot vector representation of the class, need to know how many classes there are to begin with
        # Find out how many ids there are
        for (c, f_in) in enumerate(colour_images):
            tokens = f_in.split("_")
            if tokens.__len__() < 5:
                continue
            if tokens[0] not in self.ids:
                self.ids.append(tokens[0])

            dis = float(tokens[1].replace(",", ""))
            dist = int((round(dis / 500.0) * 500.0) / 1000.0)
            if distance is not None and dist == distance:
                file_count += 1
                indices.append(c)
            elif distance is None:
                indices.append(c)

        labels = len(self.ids)

        if file_count == 0:
            file_count = (len(colour_images if (rgb or grey) else ir_images if ir else depth_images))

        print "File count: %d" % file_count

        if one_hot:
            y = np.zeros((file_count, labels), dtype=np.float32)  # Create an empty label matrix
        else:
            y = np.zeros((file_count,), dtype=np.float32)  # Create an empty label matrix

        if flatten:
            x = np.empty((file_count, height * width * num_channels), dtype=np.float32)  # Create an empty data matrix
        else:
            x = np.empty((file_count, height, width, num_channels), dtype=np.float32)  # Create an empty data matrix

        #  Find out what image streams we are reading
        if colour:
            images = colour_images
        elif ir:
            images = ir_images
        else:  # Depth
            images = depth_images

        for (idx, c) in enumerate(indices):
            f_in = images[c]
            tokens = f_in.split("_")

            f = os.path.join(self.in_file, f_in)  # Generate path to file

            if grey:
                samples = cv2.imread(f, 0)  # Read in image as greyscale
            elif rgb:
                samples = cv2.imread(f)  # or colour (3 channel)
            else:
                samples = np.genfromtxt(f, delimiter=',')

            if samples.shape[1] < 1:
                print "Error reading image file "+f_in
                continue

            sample = cv2.resize(samples, (width, height))

            if flatten:
                sample = sample.reshape(-1)
            elif not rgb:
                sample = sample.reshape((height, width, 1))

            # Deal with the other image sets
            if colour and ir:  # Read IR images if using them
                ir_img = ir_images[c]
                sample = self.__handle_csv(ir_img, sample, width, height, flatten)
            if (colour or ir) and d:  # Same for Depth
                d_img = depth_images[c]
                sample = self.__handle_csv(d_img, sample, width, height, flatten)

            x[idx] = sample

            if one_hot:
                im_id = self.ids.index(tokens[0])
                y[idx][im_id] = 1.
            else:
                y[idx] = self.ids.index(tokens[0])

            idx += 1

        print "Sanity check : first file data ----"
        print x[0]
        print "Labels:"
        print self.ids

        return [x, y]

    def __handle_csv(self, filename, sample, width, height, flatten):
        """
        IR and Depth images are stored in CSV files.
        Read them in from disk, then either append them on to the current image vector if the
        vector is flattened (1D), or stack the new image on to the bottom of current image array (if it isn't flat).

        Args:
            filename: file to fetch
            sample: current image vector
            width: width of image
            height: height of image
            flatten: are we flattening it?
        Return:
             The original image sample, with more data tacked on from a CSV file.
        """
        f = os.path.join(self.in_file, filename)
        image = np.genfromtxt(f, delimiter=',')
        image = cv2.resize(image, (width, height))

        if flatten:
            sample = np.concatenate([sample, image.reshape(-1)])
        else:
            image.reshape((height, width, 1))
            sample = np.dstack([sample, image])

        return sample
