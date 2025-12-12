# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# parameter_manager.py - manages the parameters for the current experiment
# JSON structure:
#     checkpoint_parameters
#         - model_title (String)
#         - save_model_checkpoint (int)
#     data_parameters
#         - dataset_name (String)
#         - num_buckets (int)
#         - histograms_per_image (int)
#         - color space (String)
#         - histogram saturation method (String)
#         - clip upper bound (int)
#         - projection type (String)
#         - num folds (int)
#         - num folds (String)
#         - image scale factor (float)
#     threshold_parameters
#         - blue_threshold_8bit (int)
#         - green_threshold_8bit (int)
#         - red_threshold_8bit (int)
#     model_parameters
#         - channel_size (int)
#         - kernel_size (int)
#         - stride (int)
#         - pool_size (int)
#         - pool_stride (int)
#         - convolutions_post_pool (int)
#         - convolutions_post_concat (int)
#         - includes_intermediate_linear_layer (bool)
#         - intermediate_linear_layer_features (int)
#         - p_dropout_pre_concat (float)
#         - p_dropout_post_concat (float)
#         - second_pool_layer (bool)
#         - includes_global_avg_pool (bool)
#         - stride_2_pooling (bool)
#         - pre_concat_structure_number (int)
#         - post_concat_structure_number (int)
#         - second_pool_after_second_conv (bool)
#         - uses_1x1_layer (bool)
#     learning_parameters
#         - optimizer (String)
#         - learning_rate (float)
#         - epochs (int)

import json


class ParameterManager:
    """
    Class to manage the parameters used in an experiment. Parses a JSON file to get the parameters
    """
    def __init__(self, path_parameter_json):
        """
        Initialization to read the JSON and set parameters
        :param path_parameter_json: Path to the JSON file to read parameters from
        """
        # Load the json file
        parameter_data = load_json(path_parameter_json)

        # Checkpoint parameters
        self.model_title = parameter_data['checkpoint_parameters'][0]['model_title']
        self.save_model_checkpoint = parameter_data['checkpoint_parameters'][0]['save_model_checkpoint']

        # Set the histogram parameters
        self.dataset_name = parameter_data['data_parameters'][0]['dataset_name']
        self.num_buckets = parameter_data['data_parameters'][0]['num_buckets']
        self.histograms_per_image = parameter_data['data_parameters'][0]['histograms_per_image']
        self.color_space = parameter_data['data_parameters'][0]['color_space']
        self.histogram_saturation_method = parameter_data['data_parameters'][0]['histogram_saturation_method']
        self.clip_upper_bound = parameter_data['data_parameters'][0]['clip_upper_bound']
        self.projection_type = parameter_data['data_parameters'][0]['projection_type']
        self.num_folds = parameter_data['data_parameters'][0]['num_folds']
        self.image_scale_factor = parameter_data['data_parameters'][0]['image_scale_factor']

        # Threshold parameters
        self.blue_threshold_8bit = parameter_data['threshold_parameters'][0]['blue_threshold_8bit']
        self.green_threshold_8bit = parameter_data['threshold_parameters'][0]['green_threshold_8bit']
        self.red_threshold_8bit = parameter_data['threshold_parameters'][0]['red_threshold_8bit']

        # Set the model parameters
        self.start_channel_size = parameter_data['model_parameters'][0]['start_channel_size']
        self.kernel_size = parameter_data['model_parameters'][0]['kernel_size']
        self.stride = parameter_data['model_parameters'][0]['stride']
        self.pool_kernel = parameter_data['model_parameters'][0]['pool_kernel']
        self.pool_stride = parameter_data['model_parameters'][0]['pool_stride']
        self.convolutions_post_pool = parameter_data['model_parameters'][0]['convolutions_post_pool']
        self.convolutions_post_concat = parameter_data['model_parameters'][0]['convolutions_post_concat']
        self.includes_intermediate_linear_layer = parameter_data['model_parameters'][0]['includes_intermediate_linear_layer']
        self.intermediate_linear_layer_features = parameter_data['model_parameters'][0]['intermediate_linear_layer_features']
        self.p_dropout_pre_concat = parameter_data['model_parameters'][0]['p_dropout_pre_concat']
        self.p_dropout_post_concat = parameter_data['model_parameters'][0]['p_dropout_post_concat']
        self.second_pool_layer = parameter_data['model_parameters'][0]['second_pool_layer']
        self.includes_global_avg_pool = parameter_data['model_parameters'][0]['includes_global_avg_pool']
        self.stride_2_pooling = parameter_data['model_parameters'][0]['stride_2_pooling']
        self.pre_concat_structure_number = parameter_data['model_parameters'][0]['pre_concat_structure_number']
        self.post_concat_structure_number = parameter_data['model_parameters'][0]['post_concat_structure_number']
        self.second_pool_after_second_conv = parameter_data['model_parameters'][0]['second_pool_after_second_conv']
        self.uses_1x1_layer = parameter_data['model_parameters'][0]['uses_1x1_layer']

        # Set the learning parameters
        self.scheduler = parameter_data['learning_parameters'][0]['scheduler']
        self.optimizer = parameter_data['learning_parameters'][0]['optimizer']
        self.learning_rate = parameter_data['learning_parameters'][0]['learning_rate']
        self.momentum = parameter_data['learning_parameters'][0]['momentum']
        self.epochs = parameter_data['learning_parameters'][0]['epochs']
        self.batch_size = parameter_data['learning_parameters'][0]['batch_size']
        self.random_state = parameter_data['learning_parameters'][0]['random_state']
        self.loss_function = parameter_data['learning_parameters'][0]['loss_function']


def load_json(json_path):
    """
    Loads a JSON file
    :param json_path: Path to the JSON file to load
    :return: The loaded JSON file's data
    """
    with open(json_path, 'r') as json_file:
        parameter_data = json.load(json_file)
    return parameter_data
