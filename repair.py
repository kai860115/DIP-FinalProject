import os
import argparse
import cv2
import numpy as np

from hole_filling import HF


def parse_args():
    parser = argparse.ArgumentParser(description='depthInpainting')

    parser.add_argument('--input', default='', type=str)
    parser.add_argument('--output', default='./repaired.png', type=str)
    parser.add_argument('--threshold', default=48, type=int)
    parser.add_argument('--hole_dilate_iter', default=0, type=int)
    parser.add_argument('--boundary_dilate_iter', default=1, type=int)
    parser.add_argument('--dilate_kernel_size', default=3, type=int)

    args = parser.parse_args()
    return args

def main(args):
    img = cv2.imread(args.input, flags=cv2.IMREAD_GRAYSCALE)
    hf = HF(img, threshold=args.threshold, hole_dilate_iter=args.hole_dilate_iter, boundary_dilate_iter=args.boundary_dilate_iter, dilate_kernel_size=(args.dilate_kernel_size, args.dilate_kernel_size))
    repaired= hf.repair()
    cv2.imwrite(args.output, repaired)

if __name__ == '__main__':
    args = parse_args()
    main(args)