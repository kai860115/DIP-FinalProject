import cv2
import numpy as np
from hole_filling import HF

image = cv2.imread('data/Piano.png', cv2.IMREAD_GRAYSCALE)

hf = HF(image, threshold=48, hole_dilate_iter=0, boundary_dilate_iter=1, dilate_kernel_size=(3,3))

repaired= hf.repair()

cv2.imwrite('result/repaired.png', repaired)

