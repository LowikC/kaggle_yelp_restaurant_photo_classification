# -*- coding: utf8 -*-
import numpy as np
from caffe.io import resize_image


class ImageTransformer:
    """Transform input for feeding into a Net.
    """
    def __init__(self, input_shape):
        """Create a Transformer to the input shape.

        Args:
            input_shape: (N, C, H, W) ndarray.
        """
        self.input_shape = input_shape
        self.transpose = (2, 0, 1)
        self.dimensions = input_shape[2:]
        self.channel_swap = None
        self.mean = None

    def preprocess(self, data):
        """Format input for Caffe:
        - convert to single precision
        - resize to required dimensions
        - center crop image input dimensions
        - transpose dimensions to K x H x W
        - reorder channels (for instance color to BGR) if required
        - scale raw input (e.g. from [0, 1] to [0, 255] for ImageNet models) if required
        - subtract mean if required.
        Args:
            data : (H' x W' x K) ndarray

        Returns:
            caffe_in : (K x H x W) ndarray for input to a Net
        """
        caffe_in = data.astype(np.float32, copy=False)
        in_dims = self.input_shape[2:]
        # Resize if needed.
        if caffe_in.shape[:2] != self.dimensions:
            caffe_in = resize_image(caffe_in, self.dimensions)
        # Take center crop if needed.
        if caffe_in.shape[:2] != in_dims:
            center = np.array(self.dimensions) / 2.0
            top_left = (center - np.array(in_dims) / 2.0).astype(np.int)
            bottom_right = top_left + in_dims
            caffe_in = caffe_in[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
        if self.transpose is not None:
            caffe_in = caffe_in.transpose(self.transpose)
        if self.channel_swap is not None:
            caffe_in = caffe_in[self.channel_swap, :, :]
        if self.mean is not None:
            caffe_in -= self.mean
        return caffe_in

    def set_channel_swap(self, order):
        """Set the input channel order as needed for the reference ImageNet model.
        N.B. this assumes the channels are the first dimension AFTER transpose.

        Args:
            order (tuple): the order to take the channels. (2,1,0) maps RGB to BGR for example.
        """
        if len(order) != self.input_shape[1]:
            raise Exception('Channel swap needs to have the same number of '
                            'dimensions as the input channels.')
        self.channel_swap = order

    def set_mean(self, mean):
        """Set the mean to subtract for centering the data.

        Args:
            mean : mean ndarray (input dimensional or broadcastable)
        """
        ms = mean.shape
        if mean.ndim == 1:
            # broadcast channels
            if ms[0] != self.input_shape[1]:
                raise ValueError('Mean channels incompatible with input.')
            mean = mean[:, np.newaxis, np.newaxis]
        else:
            # elementwise mean
            if len(ms) == 2:
                ms = (1,) + ms
            if len(ms) != 3:
                raise ValueError('Mean shape invalid')
            if ms != self.input_shape[1:]:
                raise ValueError('Mean shape incompatible with input shape.')
        self.mean = mean

    def set_dimensions(self, dimensions):
        """Set the dimensions of the image, before cropping it.

        Args:
            dimensions (tuple): Height, width of the image.
        """
        if dimensions[0] < self.input_shape[2]:
            raise Exception('Height must be superior or equal to the destination blob')
        if dimensions[1] < self.input_shape[3]:
            raise Exception('Width must be superior or equal to the destination blob')
        self.dimensions = dimensions
