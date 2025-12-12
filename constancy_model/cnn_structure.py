# David Treadwell
# treadwell.d@northeastern.edu
# Northeastern University
# Professor Bruce Maxwell
# cnn_structure.py - defines the structure (layers and forward pass) for the CNN

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstancyNetwork(nn.Module):
    """
    Class for the overall network used in the constancy experiments
    """
    def __init__(self, parameters):
        """
        Initialization defines the layers within the network
        """
        super(ConstancyNetwork, self).__init__()

        # Set the network parameters
        self.histograms_per_image = parameters.histograms_per_image
        self.uses_1x1_layer = parameters.uses_1x1_layer

        # Padding size
        self.padding = (2, 2)

        # Set the channel structures
        pre_concat_channels = get_pre_concat_channels(parameters.pre_concat_structure_number)
        post_concat_channels = get_post_concat_channels(parameters.post_concat_structure_number)

        # Layers of the network for each individual histogram
        self.per_hist_block = nn.ModuleList()
        for i in range(self.histograms_per_image):
            self.per_hist_block.append(_PerHistogramBlock(parameters=parameters,
                                                          padding=self.padding,
                                                          pre_concat_channels=pre_concat_channels))

        # Calculate the number of channels in the output
        concatenated_channel_count = pre_concat_channels[-1] * self.histograms_per_image

        # Use 1x1 layer if required
        if self.uses_1x1_layer:
            self.conv_1x1_post_concat = nn.Conv2d(in_channels=concatenated_channel_count,
                                                  out_channels=post_concat_channels[0],
                                                  kernel_size=1,
                                                  stride=1)

        # First post-concatenation convolution
        self.conv2d_post_1 = nn.Conv2d(in_channels=post_concat_channels[0] if self.uses_1x1_layer else concatenated_channel_count,
                                       out_channels=post_concat_channels[0],
                                       kernel_size=parameters.kernel_size,
                                       stride=2,
                                       padding=self.padding)

        # Dropout for concatenated outputs
        self.dropout_post_concat = nn.Dropout(p=parameters.p_dropout_post_concat)

        # Second convolution
        self.conv2d_post_2 = nn.Conv2d(in_channels=post_concat_channels[0],
                                       out_channels=post_concat_channels[1],
                                       kernel_size=parameters.kernel_size,
                                       stride=2,
                                       padding=self.padding)

        # Global average pooling
        self.global_average_pooling = nn.AdaptiveMaxPool2d((1, 1))

        # input size needs to be # channels from last convolution * shape
        self.flatten = nn.Flatten()

        # Calculate input features to linear layer.
        # Needs to be calculated explicitly to determine the number features after flattening
        # Cannot be smaller than 1 since doing same convolutions
        self.linear_dimensions = max(1, int(
            parameters.num_buckets
            / 2  # divided by 2 because always doing a dimension reduction with stride 2
            / 2  # Second dimensionality reduction by second stride 2 convolutional layer in pre-concat block
            / 2  # last convolution always reduces dimensionality by 2
            / 2  # first convolution after concatenating
            / 2  # final convolution of network
        ))

        # Intermediate linear layer
        self.intermediate_linear = nn.Linear(in_features=self.conv2d_post_2.out_channels,
                                             out_features=parameters.intermediate_linear_layer_features)

        # Final linear layer
        final_layer_input = self.intermediate_linear.out_features
        self.fully_connected_output = nn.Linear(in_features=final_layer_input, out_features=3)

    def forward(self, x):
        """
        Defines the forward pass for the entire constancy network
        :param x: The input (four histograms)
        :return: The three-channel mean illuminant-color estimation distributions
        """
        # Get the image histograms
        histograms = [x[:, i] for i in range(self.histograms_per_image)]

        # Run the double convolution block for each histogram
        # No activations as this happens in the block
        for i in range(len(histograms)):
            histograms[i] = self.per_hist_block[i](histograms[i])

        # Concatenate the projected convolutions
        x_concat = torch.concatenate(histograms, dim=1)

        # Use 1x1 layer if required
        if self.uses_1x1_layer:
            x_concat = self.conv_1x1_post_concat(x_concat)

        # Apply first convolution on concatenated convolution
        x_concat = F.relu(self.conv2d_post_1(x_concat))
        x_concat = self.dropout_post_concat(x_concat)

        # Apply the second convolution
        x_concat = F.relu(self.conv2d_post_2(x_concat))

        # Apply global average pooling
        x_concat = self.global_average_pooling(x_concat)

        # Flatten, apply next linear intermediate features, and get output
        x_concat = self.flatten(x_concat)
        x_concat = F.relu(self.intermediate_linear(x_concat))
        x_concat = F.sigmoid(self.fully_connected_output(x_concat))
        return x_concat


class _PerHistogramBlock(nn.Module):
    """
    Defines a standard subset of the constancy network that each histogram will separately use
    """
    def __init__(self, parameters, padding, pre_concat_channels):
        """
        Initializes the network layer subset
        :param parameters:          The set of parameters to use in model training
        :param padding:             The padding to enforce same convolutions
        :param pre_concat_channels: The channel structure for the pre-concatenation block
        """
        super(_PerHistogramBlock, self).__init__()
        self.second_pool_after_second_conv = parameters.second_pool_after_second_conv

        # First convolution
        self.conv2d_pre_1 = nn.Conv2d(in_channels=pre_concat_channels[0],
                                      out_channels=pre_concat_channels[1],
                                      kernel_size=parameters.kernel_size,
                                      stride=2,
                                      padding=padding)

        # Second convolution
        self.conv2d_pre_2 = nn.Conv2d(in_channels=pre_concat_channels[1],
                                      out_channels=pre_concat_channels[2],
                                      kernel_size=parameters.kernel_size,
                                      stride=2 if self.second_pool_after_second_conv else 1,
                                      padding=padding)

        # Add third convolution
        self.conv2d_pre_3 = nn.Conv2d(in_channels=pre_concat_channels[2],
                                      out_channels=pre_concat_channels[3],
                                      kernel_size=parameters.kernel_size,
                                      stride=1 if self.second_pool_after_second_conv else 2,
                                      padding=padding)

        # Dropout layer for before the final convolution
        self.dropout = nn.Dropout(p=parameters.p_dropout_pre_concat)

        # Add final convolution layer
        self.conv2d_pre_4 = nn.Conv2d(in_channels=pre_concat_channels[3],
                                      out_channels=pre_concat_channels[4],
                                      kernel_size=parameters.kernel_size,
                                      stride=2,
                                      padding=padding)

    def forward(self, x):
        """
        Defines the forward pass for the network block
        :param x: The input (will be a single histogram)
        :return: The result of the convolution stack
        """
        # Convolution
        x = F.relu(self.conv2d_pre_1(x))

        # Second convolution
        x = F.relu(self.conv2d_pre_2(x))

        # Apply an additional convolution
        x = F.relu(self.conv2d_pre_3(x))

        # Final convolution
        x = self.dropout(x)
        x = F.relu(self.conv2d_pre_4(x))
        return x


def get_pre_concat_channels(pre_concat_structure_number):
    """
    Get the number of pre-concatenation channels based on a number (for tuning purposes)
    :param pre_concat_structure_number: The number (index) of channels to use
    :return: An array of channel counts
    """
    channel_structure = [
        [1, 8, 16, 16, 32],
        [1, 8, 16, 32, 32],
        [1, 16, 32, 32, 32],
        [1, 16, 16, 32, 32],
        [1, 16, 16, 16, 32],
        [1, 32, 32, 32, 32],
    ]
    return channel_structure[pre_concat_structure_number]


def get_post_concat_channels(post_concat_structure_number):
    """
    Get the number of post-concatenation channels based on a number (for tuning purposes)
    :param post_concat_structure_number: The number (index) of channels to use
    :return: An array of channel counts
    """
    channel_structure = [
        [64, 128],
        [64, 64],
        [128, 128],
    ]
    return channel_structure[post_concat_structure_number]
