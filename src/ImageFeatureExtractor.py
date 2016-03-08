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


def compute_features(images, net, feature_layers=None):
    """ Compute features and output probabilities on each image.

    Args:
        images: List of full paths to the images to process.
        net: Caffe.Classifier to process the image.
        feature_layer: Name of the feature layers to extract.

    Returns:
        dict:  
        probs is a (N, K) array (N: number of samples, K: number of classes).
        features is a (N, F) array (F: number of features)
    """
    features = dict()
    for feature_layer in feature_layers:
        if feature_layer in net.blobs:
            features[feature_layer] = np.zeros((len(images), net.blobs[feature_layer].channels))

    if not features:
        return
        
    batch_size = net.blobs['data'].num

    n_images = len(images)
    im_per_batch = batch_size/n_crops
    n_batches = int(math.ceil(n_images * 1.0/im_per_batch))

    print("The network will process {b_size} images in parallel.".format(b_size=batch_size))
    print("We load {n_im} images per batch".format(n_im=im_per_batch))

    for b in xrange(n_batches):
        try:
            # Load a batch
            offset = b * im_per_batch
            actual_size = min(im_per_batch, n_images - offset)
            input_images = [caffe.io.load_image(images[offset + i]) for i in xrange(actual_size)]
        except Exception as err:
            print("Can't load image : {e}".format(e=err.message))
        else:
            net.predict(input_images, oversample=False)
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
                        help='Use GPU')
    parser.add_argument('--feature_layer', type=str, default='',
                        help="Layer to use to extract features (usually: 'fc7'). You can put several layer like 'fc6 fc7'. ")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_args()

    caffe.set_mode_cpu()
    net = caffe.Classifier(args.deploy, args.weights,
                           mean=np.load(args.mean).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
    if args.gpu:
        caffe.set_mode_gpu()

    images = load_images_list(args.input, args.image_root)
    features = compute_features_and_probs(images, net, args.feature_layer.split())

    # Save computed arrays
    _, basename = os.path.split(args.input)
    basename, _ = os.path.splitext(basename)
    for layer_name, features_array in features.items():
        np.save(os.path.join(args.output, basename + '_' + layer_name + '.npy'), features_array)

