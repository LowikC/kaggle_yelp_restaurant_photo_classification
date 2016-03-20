import numpy as np
import os
from PIL import Image
import caffe
import math
import argparse


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


def compute_features(images, net, image_mean, feature_layers):
    """ Compute features and output probabilities on each image.

    Args:
        images (list): List of full paths to the images to process.
        image_mean (numpy.array): Mean value per channel of the training set.
        net (caffe.Net): CNN to process the image.
        feature_layer (list): Names of the feature layers to extract.

    Returns:
        A dict {layer_name: data} where data is a (N, F) numpy array.
        (N: number of samples, F: number of features).
    """

    # Create empty arrays for each feature
    features = dict()
    for feature_layer in feature_layers:
        if feature_layer in net.blobs:
            feature_blob = net.blobs[feature_layer]
            n_features = feature_blob.width * feature_blob.height * feature_blob.channels
            features[feature_layer] = np.zeros((len(images), n_features))

    if not features:
        return
        
    batch_size = net.blobs['data'].num
    n_images = len(images)
    n_batches = int(math.ceil(n_images * 1.0/batch_size))

    print("The network will process {b_size} images in parallel.".format(b_size=batch_size))
    print("We load {n_im} images per batch".format(n_im=batch_size))
    for layer_name, _ in features.items():
        print("We extract features on layer {name}".format(name=layer_name))


    # Define the preprocessor
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', image_mean) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]]

    # Process per batch
    n, c, w, h = net.blobs['data'].data.shape
    for b in xrange(n_batches):
        offset = b * batch_size
        actual_size = min(batch_size, n_images - offset)

        net.blobs['data'].reshape(actual_size, c, w, h)
        net.blobs['data'].data[...] = \
            map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)),
                images[offset:offset+actual_size])

        net.forward()

        for layer_name, features_array in features.items():
            features_array[offset:offset+actual_size, :] = \
                net.blobs[layer_name].data[:actual_size].reshape((actual_size, -1))
        print("Batch {b}/{n_b} done".format(b=b, n_b=n_batches))

    return features


def get_command_args():
    parser = argparse.ArgumentParser(description='Extract CNN features.')
    parser.add_argument('--input', '-i', type=str,
                        help=('path to a file containing the image paths.'
                              ' You may use the same file as for Caffe training/testing'))
    parser.add_argument('--image_root', type=str,
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
    parser.add_argument('--gpu', action="store_true",
                        help='Use GPU if defined, otherwise CPU only.')
    parser.add_argument('--features_layers', nargs='*', dest='features_layers', action='append',
                        help="Layer to use to extract features. You can put several layers:"
                             "--features_layers fc7 fc6")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_args()

    caffe.set_mode_cpu()
    net = caffe.Net(args.deploy, args.weights, caffe.TEST)

    if args.gpu:
        caffe.set_mode_gpu()

    images = load_images_list(args.input, args.image_root)
    image_mean = np.load(args.mean).mean(1).mean(1)
    features_layers = [el for elements in args.features_layers for el in elements]
    features = compute_features(images, net, image_mean, features_layers)

    # Save computed arrays
    _, basename = os.path.split(args.input)
    basename, _ = os.path.splitext(basename)
    for layer_name, features_array in features.items():
        np.save(os.path.join(args.output, basename + '_' + layer_name + '.npy'), features_array)


