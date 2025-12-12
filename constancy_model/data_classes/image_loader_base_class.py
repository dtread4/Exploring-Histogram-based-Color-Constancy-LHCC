# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# image_loader_base_class.py - Abstract base class to define how each dataset's image loader should be implemented

from abc import ABC, abstractmethod
import numpy as np
import os
import pandas as pd
import cv2


class ImageLoader(ABC):
    """
    Class to define shared methods and data for each dataset's image loader
    """
    def __init__(self, black_level, saturation_value, validation_fold_number, train_images_path, valid_images_path,
                 train_gt_path, valid_gt_path, cross_fold_path, image_scale_factor):
        # Set basic values
        self.black_level = black_level
        self.saturation_value = saturation_value
        self.upper_bound = self.saturation_value - self.black_level
        self.validation_fold_number = validation_fold_number
        self.image_scale_factor = image_scale_factor

        # Set data paths
        self.train_images_path = train_images_path
        self.valid_images_path = valid_images_path
        self.train_gt_path = train_gt_path
        self.valid_gt_path = valid_gt_path

        # Set cross fold labels
        if cross_fold_path is not None:
            self.fold_labels = self.load_cross_folds(cross_fold_path)

        # Pre-set variables that will be created later
        self.train_gt_df = None
        self.valid_gt_df = None

    def __len__(self):
        """
        Note that len function will not work as expected until a child class initializes the ground truth DataFrames
        """
        return len(self.train_gt_df.index) + len(self.valid_gt_df.index)

    @abstractmethod
    def load_image(self, train_data, image_index):
        """
        Abstract method to enforce the inclusion of a method to load an image.
        Preprocessing should be handled within this method
        :param train_data:  True if loading a train image, False if not
        :param image_index: The index (within train/valid DataFrame) of the image to load
        :return: The image ready to use
        """
        pass

    @abstractmethod
    def load_ground_truth_dataframe(self):
        """
        Abstract method to enforce the inclusion of a method to load the ground truth DataFrame
        Some specifics differ depending on how ground truth is stored
        :return: None
        """
        pass

    def get_valid_image_names(self):
        """
        Returns a list of all images in the validation set
        :return: All validation set image names
        """
        return self.valid_gt_df['image']

    def get_image_upper_bound(self, train_data, image_index):
        """
        Returns the upper bound (saturation value - black level) for an image
        Most datasets use the same upper bound, so train data and index are unused by default.
        Provided as a parameters for overriding and use in dataset-specific tasks
        :param train_data:  Whether image is from training data or not
        :param image_index: The index of the image to get the upper bound for
        :return: The upper bound for the dataset
        """
        return self.upper_bound

    def load_cross_folds(self, cross_fold_path):
        """
        Takes a text file containing the fold labels and puts them into an array
        :param cross_fold_path: Path to the cross fold validation labels file
        :return: Numpy array containing the folds
        """
        fold_labels = []
        with open(cross_fold_path, 'r') as file:
            for line in file.readlines():
                fold_labels.append(int(line.strip()))
        return np.array(fold_labels)

    def load_image_names(self, data_path):
        """
        Gets list of all file names in the image folder and puts them into a single numpy array
        :param data_path: The path to the image data directory
        :return: Numpy array containing file names
        """
        image_names = np.array(os.listdir(data_path))
        return image_names

    def create_ground_truth_dataframes_from_folds(self, image_names, red_gt, green_gt, blue_gt):
        """
        Creates a ground truth DataFrame with a standard format
        :param image_names: Array of image names
        :param red_gt:      Array of red ground truth values
        :param green_gt:    Array of green ground truth values
        :param blue_gt:     Array of blue ground truth values
        :return: None
        """
        data = {
            "image": image_names,
            "mean_r": red_gt,
            "mean_g": green_gt,
            "mean_b": blue_gt,
            "validation_fold": self.fold_labels
        }
        ground_truth_df = pd.DataFrame(data)

        # Set train and valid DataFrames
        self.train_gt_df = ground_truth_df[ground_truth_df['validation_fold'] != self.validation_fold_number]
        self.valid_gt_df = ground_truth_df[ground_truth_df['validation_fold'] == self.validation_fold_number]

    def create_ground_truth_dataframes_from_presplit_data(self):
        """
        Loads the ground truth data into a train and test array
        :return: None
        """
        # Load the csv files into a DataFrame
        self.train_gt_df = pd.read_csv(self.train_gt_path)
        self.valid_gt_df = pd.read_csv(self.valid_gt_path)

    def load_original_image(self, train_data, image_index):
        """
        Loads an image based on its index, without performing preprocessing unless shared by all datasets (e.g. resize)
        :param train_data:  True if a training image, False if a validation image
        :param image_index: Index of image within train/validation DataFrame
        :return: The original image without preprocessing
        """
        # Get the image name from the correct Dataframe
        image_name = self.get_image_name_from_index(train_data, image_index)

        # Create full image path
        if train_data:
            full_image_path = self.train_images_path + os.sep + image_name
        else:
            full_image_path = self.valid_images_path + os.sep + image_name

        # Load image with OpenCV
        image = cv2.imread(full_image_path, cv2.IMREAD_UNCHANGED)

        # Adjust black level and scale image to appropriate bit depth
        image = self.adjust_black_level(image, self.black_level, self.upper_bound)
        return image

    def resize_image(self, image):
        """
        Resize image to specified scale factor (+ 0.5 to round up decimals if >= x.5)
        Not applied during basic loading because some operations specific to datasets require cropping, which would
        be more complicated to perform after resizing than before
        :param image: The image to resize
        :return: The resized image
        """
        if self.image_scale_factor != 1:
            new_height = int(image.shape[0] * self.image_scale_factor + 0.5)
            new_width = int(image.shape[1] * self.image_scale_factor + 0.5)
            image = cv2.resize(image, (new_width, new_height), cv2.INTER_AREA)
        return image

    def get_image_name_from_index(self, train_data, image_index):
        """
        Gets the image name based on if it is a train or validation image, and its index within its DataFrame
        :param train_data:  True if a training image, False if a validation image
        :param image_index: Index of image within train/validation DataFrame
        :return: The name of the image
        """
        # Get the image name from the correct Dataframe
        if train_data:
            image_name = self.train_gt_df.iloc[image_index, 0]
        else:
            image_name = self.valid_gt_df.iloc[image_index, 0]
        return image_name

    def adjust_black_level(self, image, black_level, upper_bound):
        """
        Reduces an image's black level to zero
        :param image:       The image to reduce the black level of
        :param black_level: The black level to subtract away
        :param upper_bound: The upper bound of the image
        :return: The image with its black level normalized
        """
        org_image_type = image.dtype
        image = image.astype(np.int32)
        image = np.clip(image - black_level, 0, upper_bound)
        return image.astype(org_image_type)

    def scale_image(self, image, org_upper_bound, new_upper_bound):
        """
        Scales an image to a new range; assumes the original image had a black point of 0
        Assumes that the new upper bound is less than a 16-bit unsigned integer value
        :param image:           The image to scale
        :param org_upper_bound: The original image upper bound (may differ from maximum image value)
        :param new_upper_bound: The new image upper bound
        :return: The image scaled to the appropriate bit depth
        """
        # Scale image down to [0, 1] range then multiply to new upper bound range
        image = np.clip(image.astype(np.float32) / org_upper_bound, 0, 1)
        image *= new_upper_bound

        # Set type appropriately for new image range
        image = image.astype(np.uint8) if new_upper_bound <= (2**8 - 1) else image.astype(np.uint16)
        return image

