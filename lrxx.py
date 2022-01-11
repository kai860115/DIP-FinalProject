import os
import argparse
import cv2
import numpy as np

from lrl0 import LRL0
from lrl0phi import LRL0PHI
from tnnr import TNNR
from lrtv import LRTV

def parse_args():
    parser = argparse.ArgumentParser(description='depthInpainting')

    parser.add_argument('--method', default='LRTV', type=str, choices=['LRTV', 'LRL0', 'LRL0PHI', 'repair'])
    parser.add_argument('--depthImage', default='', type=str)
    parser.add_argument('--mask', default='', type=str)
    parser.add_argument('--outputPath', default='./inpainting', type=str)
    parser.add_argument('--data', default='Teddy', type=str)
    parser.add_argument('--initImage', default='', type=str)
    parser.add_argument('--K', default=3, type=int)
    parser.add_argument('--lambda_L0', default=30, type=int)
    parser.add_argument('--MaxIterCnt', default=30, type=int)

    args = parser.parse_args()
    return args

def main(args):
    disparityMissing = cv2.imread(args.depthImage, flags=cv2.IMREAD_GRAYSCALE)
    orig = np.copy(disparityMissing)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    disparityMissing = disparityMissing * (mask / 255)
    denoised = cv2.imread(args.initImage, cv2.IMREAD_GRAYSCALE)
    # denoised = TNNR(disparityMissing, mask, 9, 9, 1e-2)
    if args.method == 'LRTV':
        inpaintingPath = f"{args.outputPath}/LRTV_result/{args.data}"
        os.makedirs(inpaintingPath, exist_ok=True)
        lrtv = LRTV(disparityMissing, mask, denoised, 1.2, 0.1, 40, 10)
        result = lrtv.compute()
        M = lrtv.getM()
        Y = lrtv.getY()
        output = result.astype(np.uint8)
        cv2.imwrite(f'{inpaintingPath}/lrtv.png', output)
        cv2.imwrite(f'{inpaintingPath}/M.png', M)
        cv2.imwrite(f'{inpaintingPath}/Y.png', Y)
        
    elif args.method == 'LRL0':
        inpaintingPath = f"{args.outputPath}/LRL0_result/{args.K}_{args.lambda_L0}/{args.data}"
        os.makedirs(inpaintingPath, exist_ok=True)
        lrl0 = LRL0(disparityMissing, mask, denoised, 1.2, 0.1, args.lambda_L0, 10)
        result = lrl0.compute(args.K, args.MaxIterCnt, inpaintingPath, orig)
        output = result.astype(np.uint8)
        cv2.imwrite(f'{inpaintingPath}/lrl0.png', output)
        
    elif args.method == 'LRL0PHI':
        inpaintingPath = f"{args.outputPath}/LRL0PHI_result/{args.K}_{args.lambda_L0}/{args.data}"
        os.makedirs(inpaintingPath, exist_ok=True)
        lrl0phi = LRL0PHI(disparityMissing, mask, denoised, 1.2, 0.1, args.lambda_L0, 10, 0.75)
        result = lrl0phi.compute(args.K, args.MaxIterCnt, inpaintingPath, orig)
        output = result.astype(np.uint8)
        cv2.imwrite(f'{inpaintingPath}/lrl0phi.png', output)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
