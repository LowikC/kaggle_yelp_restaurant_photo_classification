import numpy as np
import cv2
from collections import namedtuple

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])

class ImageResizer(object):

    def __init__(self):
        self._margin = 0
        self._foreground = Rect(0, 0, 0, 0)
        self._max_roi = Rect(0, 0, 0, 0)

    def resize_crop(self, image, size=(256, 256), interpolation=cv2.INTER_AREA, margin=0.1):
        """ Resize the image, to a given size.
        It keeps the aspect ratio of the image by cropping the image.
        :param image: Input image, should be a np.array with uint8 type.
        :param size: Output size of the image.
        :param interpolation: Interpolation method used for the resize.
        :param find_foreground: If True, we search for the foreground, and resize only this part.
        :param margin: If find_foreground is used, we add a margin around the image.
        :return: The resized image.
        """
        self._margin = margin
        self._foreground = Rect(0, 0, image.shape[1], image.shape[0])
        self._find_maximal_square_roi(image, size)
        return self._resize_max_roi(image, size, interpolation)

    def resize_random_bg(self, image, size=(256, 256), interpolation=cv2.INTER_AREA, margin=0.1):
        """ Resize the image, to a given size.
        It keeps the aspect ratio of the image by filling the missing parts with random noise.
        :param image: Input image, should be a np.array with uint8 type.
        :param size: Output size of the image.
        :param interpolation: Interpolation method used for the resize.
        :param margin: If find_foreground is used, we add a margin around the image.
        :return: The resized image.
        """
        self._margin = margin
        self._foreground = Rect(0, 0, image.shape[1], image.shape[0])
        self._find_maximal_roi(image, size)
        return self._resize_max_roi_w_bg(image, size, interpolation)

    def resize_squash(self, image, size=(256, 256), interpolation=cv2.INTER_AREA, margin=0.1):
        """ Resize the image, to a given size.
        It doesn't keeps the aspect ratio of the original image, so output may seems distorted.
        :param image: Input image, should be a np.array with uint8 type.
        :param size: Output size of the image.
        :param interpolation: Interpolation method used for the resize.
        :param find_foreground: If True, we search for the foreground, and resize only this part.
        :param margin: If find_foreground is used, we add a margin around the image.
        :return: The resized image.
        """
        self._margin = margin
        self._foreground = Rect(0, 0, image.shape[1], image.shape[0])
        self._find_maximal_roi(image, size)
        return self._resize_max_roi(image, size, interpolation)

    def _find_maximal_square_roi(self, image, size):
        assert size[0] == size[1]

        h, w, _ = image.shape

        dw = self._foreground.w * self._margin
        dh = self._foreground.h * self._margin

        centerx = self._foreground.x + self._foreground.w/2
        centery = self._foreground.y + self._foreground.h/2

        dcx_right = w - centerx
        dcy_bottom = h - centery

        max_half = max(self._foreground.w/2 + dw, self._foreground.h/2 + dh)
        half_w = min(dcx_right, centerx, max_half)
        half_h = min(dcy_bottom, centery, max_half)
        half_side = min(half_h, half_w)

        self._max_roi = Rect(x=centerx - half_side, y=centery - half_side, w=2 * half_side, h=2 * half_side)

    def _find_maximal_roi(self, image, size):
        assert size[0] == size[1]

        h, w, _ = image.shape

        dw = self._foreground.w * self._margin
        dh = self._foreground.h * self._margin

        centerx = self._foreground.x + self._foreground.w/2
        centery = self._foreground.y + self._foreground.h/2

        dcx_right = w - centerx
        dcy_bottom = h - centery

        max_hwidth = min(dcx_right, centerx)
        max_hheight = min(dcy_bottom, centery)

        if self._foreground.w < self._foreground.h:
            half_h = min(self._foreground.h/2 + dh, max_hheight)
            half_w = min(max_hwidth, half_h)
        else:
            half_w = min(self._foreground.w/2 + dw, max_hwidth)
            half_h = min(max_hheight, half_w)

        self._max_roi = Rect(x=centerx - half_w, y=centery - half_h, w=2 * half_w, h=2 * half_h)
        if self._max_roi.h == 0 or self._max_roi.w == 0:
            self._max_roi = Rect(x=0, y=0, w=image.shape[1], h=image.shape[1])

    def _resize_max_roi(self, image, size, interpolation):
        im_crop = image[self._max_roi.y:self._max_roi.y + self._max_roi.h,\
                        self._max_roi.x:self._max_roi.x + self._max_roi.w, :]
        return cv2.resize(im_crop, dsize=size, interpolation=interpolation)


    def _resize_max_roi_w_bg(self, image, size, interpolation):
        im_crop = image[self._max_roi.y:self._max_roi.y + self._max_roi.h,\
                        self._max_roi.x:self._max_roi.x + self._max_roi.w, :]

        target_aspected_ratio = (1.0 * size[0])/(size[1])
        original_aspect_ratio = (1.0 * im_crop.shape[1])/(im_crop.shape[0])

        # Compute the targeted size, keeping aspect ratio, and included in size
        if target_aspected_ratio > original_aspect_ratio:
            target_width = int(size[0] * original_aspect_ratio)
            target_size = (target_width, size[1])
        else:
            target_height = int(size[1] / original_aspect_ratio)
            target_size = (size[0], target_height)

        # Resize the image to the targeted size, and put it inside random noise.
        im_w_bg = np.random.randint(0, 255, (size[0], size[1], im_crop.shape[2])).astype('uint8')
        dh = (size[1] - target_size[1])/2
        dw = (size[0] - target_size[0])/2
        im_w_bg[dh:dh+target_size[1], dw:dw+target_size[0], :] = cv2.resize(im_crop, dsize=target_size, interpolation=interpolation)

        return im_w_bg


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Resize an image')
    parser.add_argument('--input', type=str, help='Source image path')
    parser.add_argument('--output', type=str, help="Destination image path")
    parser.add_argument('--size', type=tuple, default=(256, 256), help='Size of the resize image.')
    parser.add_argument('--method', type=str, default='fill', choices=['crop', 'fill', 'squash'], help='Resize method. crop or fill with random noise.')
    args = parser.parse_args()

    resizer = ImageResizer()

    src = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if args.method == 'crop':
        dst = resizer.resize_crop(src, args.size)
    elif args.method == 'fill':
        dst = resizer.resize_random_bg(src, args.size, interpolation=cv2.INTER_LINEAR)
    elif args.method == 'squash':
        dst = resizer.resize_squash(src, args.size, interpolation=cv2.INTER_LINEAR)
    else:
        sys.exit(1)

    cv2.imwrite(args.output, dst)