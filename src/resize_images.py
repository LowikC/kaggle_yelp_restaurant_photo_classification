from ImageResizer import ImageResizer
import cv2
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rzsize a list of images.')
    
    parser.add_argument('--list', type=str,
                        help='List of images paths')
    parser.add_argument('--out', type=str,
                        help='Output directory')
    parser.add_argument('--size', type=tuple,
                        default=(256, 256), help='Size of the resize image.')
    parser.add_argument('--method', type=str, 
                        default='fill', choices=['crop', 'fill', 'squash'],
                        help='Resize method. crop or fill with random noise.')


    args = parser.parse_args()

    resizer = ImageResizer()

    with open(args.list, 'r') as lfile:
        for line in lfile:
            try:
                src_path = line.strip()
                src = cv2.imread(src_path)
                
                if args.method == 'crop':
                    dst = resizer.resize_crop(src, args.size, interpolation=cv2.INTER_LINEAR)
                elif args.method == 'fill':
                    dst = resizer.resize_random_bg(src, args.size, interpolation=cv2.INTER_LINEAR)
                elif args.method == 'squash':
                    dst = resizer.resize_squash(src, args.size, interpolation=cv2.INTER_LINEAR)
                    
                _, basename = os.path.split(src_path)
                dst_path = os.path.join(args.out, basename)
                cv2.imwrite(dst_path, dst)
            except Exception as err:
                print("Can't resize image {im}: {err}".format(im=src_path, err=err))
                
