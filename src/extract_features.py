import numpy as np
import os
import sys
from PIL import Image
import caffe
import cv2
import math
import argparse
from ImageTransformer import ImageTransformer


def load_images_list(caffe_file_path, images_root):
    """ Returns the full path of all images in the Caffe list file.

    Args:
        caffe_file_path: Path to the Caffe list file.
        images_root: Path to the directory containing the images.
    Returns:
        List of full path to every images.
    """
    images_full_paths = []
    with open(caffe_file_path, 'r') as file:
        for line in file:
            image_name = line.split()[0]
            full_path = os.path.join(images_root, image_name)
            images_full_paths.append(full_path)
    return images_full_paths


def update_progress(progress, max_lenght=15):
    ratio = max_lenght/100.0
    progress_length = int(progress * ratio)
    progress_bar = '#' * progress_length + ' ' * (max_lenght - progress_length)
    sys.stdout.write('\r[{bar}] {p:.2f}%'.format(bar=progress_bar, p=progress))
    sys.stdout.flush()


def compute_features(images, net, transformer, feature_layers, batch_size, input_layer):
    """ Compute features and output probabilities on each image.

    Args:
        images (list): List of full paths to the images to process.
        net (caffe.Net): The CNN
        transformer (caffe.io.Transformer): The data transformer
        feature_layer (list): Names of the feature layers to extract.
        batch_size (int): Number of images to process per batch.

    Returns:
        A dict {layer_name: data} where data is a (N, F) numpy array.
        (N: number of samples, F: number of features).
    """

    # Create empty arrays for each feature
    # Each array has a shape NxCxHxW
    features = dict()
    for feature_layer in feature_layers:
        if feature_layer in net.blobs:
            feature_blob = net.blobs[feature_layer]
            shape = (len(images), feature_blob.channels, feature_blob.height, feature_blob.width)
            features[feature_layer] = np.zeros(shape, dtype=np.float32)

    if not features:
        return

    n_images = len(images)
    n_batches = int(math.ceil(n_images * 1.0/batch_size))

    print("The network will process {b_size} images in parallel.".format(b_size=batch_size))
    print("There are {total} images, so there will be {n} batches.".format(total=len(images),
                                                                           n=n_batches))
    for layer_name, _ in features.items():
        print("We extract features on layer {name}".format(name=layer_name))

    # Process per batch
    n, c, w, h = net.blobs[input_layer].data.shape
    for b in xrange(n_batches):
        offset = b * batch_size
        actual_size = min(batch_size, n_images - offset)
        net.blobs[input_layer].reshape(actual_size, c, w, h)
        net.blobs[input_layer].data[...] = \
            map(lambda x: transformer.preprocess(cv2.imread(x)), images[offset:offset+actual_size])

        net.forward()

        for layer_name, features_array in features.items():
            data = net.blobs[layer_name].data
            expected_shape = features_array[offset:offset+actual_size, ...].shape
            features_array[offset:offset+actual_size, ...] = data.reshape(expected_shape)

        update_progress((100.0 * b)/n_batches)

    return features


def get_command_args():
    parser = argparse.ArgumentParser(description='Extract CNN features.')
    parser.add_argument('--input', '-i', type=str,
                        help=('path to a file containing the image paths.'
                              ' You may use the same file as for Caffe training/testing'))
    parser.add_argument('--image_root', type=str, default='',
                        help=('path to the root directory containing images.'
                              'It will be appended before the image name found in the input file.'))
    parser.add_argument('--output', '-o', type=str,
                        help='path to the output directory')
    parser.add_argument('--weights', type=str,
                        help="Path to the CNN weights file (.caffemodel)")
    parser.add_argument('--deploy', type=str,
                        help='Path to the CNN deploy file (.prototxt). To speed-up the processing,'
                             'put a large batch size.')
    parser.add_argument('--mean', type=str,
                        help='Path to the CNN mean file (.npy)')
    parser.add_argument('--image_dim', type=tuple, default=(256, 256),
                        help='Dimension of the images used to train the CNN.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images processed in a batch.')
    parser.add_argument('--gpu', action="store_true",
                        help='Use GPU')
    parser.add_argument('--input_layer', type=str, default='data',
                        help='Name of the input layer.')
    parser.add_argument('--features_layers', nargs='*', dest='features_layers', action='append',
                        help="Layer to use to extract features. You can put several layers:"
                             "--features_layers fc7 fc6")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_args()
    input_layer = args.input_layer

    # Load the Network.
    caffe.set_mode_cpu()
    net = caffe.Net(args.deploy, args.weights, caffe.TEST)
    if args.gpu:
        caffe.set_mode_gpu()

    # Define the preprocessor
    image_mean = np.load(args.mean).mean(1).mean(1)
    transformer = ImageTransformer(net.blobs[input_layer].data.shape)
    transformer.set_mean(image_mean)
    transformer.set_dimensions(args.image_dim)

    images = load_images_list(args.input, args.image_root)
    features_layers = [el for elements in args.features_layers for el in elements]
    features = compute_features(images, net, transformer, features_layers, args.batch_size, input_layer)

    # Save computed arrays
    _, basename = os.path.split(args.input)
    basename, _ = os.path.splitext(basename)
    for layer_name, features_array in features.items():
        safe_name = layer_name.replace('/', '_')
        np.save(os.path.join(args.output, basename + '_' + safe_name + '.npy'), features_array)
