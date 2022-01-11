import os
import argparse
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='depthInpainting')

    parser.add_argument('--input', default='', type=str)
    parser.add_argument('--output_mask', default='./mask.png', type=str)
    parser.add_argument('--output_missing', default='./missing.png', type=str)
    parser.add_argument('--missing_rate', default=0.5, type=float)

    args = parser.parse_args()
    return args

def main(args):
    img = cv2.imread(args.input, flags=cv2.IMREAD_GRAYSCALE)
    mask = (np.random.uniform(0, 1, size=img.shape) > args.missing_rate).astype(np.uint8)
    missing = img * mask

    os.makedirs(os.path.dirname(args.output_mask), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_missing), exist_ok=True)

    cv2.imwrite(args.output_mask, mask * 255)
    cv2.imwrite(args.output_missing, missing)


if __name__ == '__main__':
    args = parse_args()
    main(args)