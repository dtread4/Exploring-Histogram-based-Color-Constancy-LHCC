# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# simplecube_image_loader.py - Dataset image loader for SimpleCube++
# 
# Please refer to usage in class description and README

from constancy_model.data_classes.image_loader_base_class import ImageLoader
from constancy_model.parameter_manager import load_json
from abc import ABC
import os


class SimpleCubeLoader(ImageLoader, ABC):
    """
    Class to load SimpleCube++ images

    Data available at: https://github.com/Visillect/CubePlusPlus?tab=readme-ov-file
    Data will be downloaded pre-sorted into train and test sets. Just extract from .zip file
    """
    def __init__(self, validation_fold_number, image_scale_factor):
        # Set shared variables
        black_level = 2048
        saturation_value = 2**14 - 1

        # Get paths for dataset
        dataset_name = "SimpleCube++"
        data_file_path = "constancy_model" + os.sep + "data_classes" + os.sep + "data_paths.json"
        dataset_paths = load_json(data_file_path)[dataset_name][0]['data_path']
        train_images_path = dataset_paths + os.sep + "train" + os.sep + "PNG"
        valid_images_path = dataset_paths + os.sep + "test" + os.sep + "PNG"
        train_gt_path = dataset_paths + os.sep + "train" + os.sep + "gt.csv"
        valid_gt_path = dataset_paths + os.sep + "test" + os.sep + "gt.csv"
        cross_fold_path = None

        # Initialize parent class
        super().__init__(black_level, saturation_value, validation_fold_number, train_images_path, valid_images_path,
                         train_gt_path, valid_gt_path, cross_fold_path, image_scale_factor)

        # Set ground truth dataframe
        self.load_ground_truth_dataframe()

    def load_ground_truth_dataframe(self):
        """
        Loads the ground truth data into a train and test array
        :return: None
        """
        # Load the csv files into a DataFrame
        self.create_ground_truth_dataframes_from_presplit_data()

        # Add file type to both DataFrames
        file_type = ".png"
        self.train_gt_df['image'] = self.train_gt_df['image'] + file_type
        self.valid_gt_df['image'] = self.valid_gt_df['image'] + file_type

    def load_image(self, train_data, image_index):
        """
        Loads an image based on its index, with preprocessing
        :param train_data:  True if loading a train image, False if not
        :param image_index: The index (within train/valid DataFrame) of the image to load
        :return: The image ready to use
        """
        # Get the original image
        original_image = self.load_original_image(train_data, image_index)

        # Resize if necessary
        resized_image = self.resize_image(original_image)
        return resized_image
