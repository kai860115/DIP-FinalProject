import cv2
import numpy as np

def norm_tv(img):
    kernelx_plus = np.array([[0.0, -1.0, 1.0]])
    kernelx_minus = np.array([[-1.0, 1.0, 0.0]])
    kernely_plus = np.array([[0.0], [-1.0], [1.0]])
    kernely_minus = np.array([[-1.0], [1.0], [0.0]])

    grad_x_plus = cv2.filter2D(img, -1, kernelx_plus)
    grad_x_minus = cv2.filter2D(img, -1, kernelx_minus)
    grad_y_plus = cv2.filter2D(img, -1, kernely_plus)
    grad_y_minux = cv2.filter2D(img, -1, kernely_minus)

    U_x = np.power((grad_x_minus + grad_x_plus) / 2, 2.0)
    U_y = np.power((grad_y_minux + grad_y_plus) / 2, 2.0)

    grad = np.sqrt(U_x + U_y)

    return np.sum(grad)

def PSNR(Xfull, Xrecover, mask):
    Xfull = Xfull.astype(np.float64)
    Xrecover = Xrecover.astype(np.float64)
    mask = mask.astype(np.float64)

    Xrecover = np.maximum(Xrecover, 0.0)
    Xrecover = np.minimum(Xrecover, 255.0)

    MSE = 0.0
    missing = (255 - mask) / 255.0
    diff = Xfull - Xrecover
    result = diff * missing

    MSE = np.linalg.norm(result)
    MSE = MSE ** 2

    nnz = np.count_nonzero(missing)

    MSE = MSE / nnz

    psnr = 10 * np.log10(255 * 255 / MSE)
    return psnr