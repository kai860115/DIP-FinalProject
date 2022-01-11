import cv2
import numpy as np
from tqdm import tqdm

def traceNorm(X):
    _, W, _ = np.linalg.svd(X)
    return np.sum(W)

def APGL(A, B, X, M, mask, eps, lambda_):
    AB = A.T @ B
    BA = B.T @ A

    W = X.shape[1]
    H = X.shape[0]

    lastX = np.copy(X)
    Y = np.copy(lastX)

    tlast = 1.0
    t = 1.0
    XX = np.copy(X)
    known = mask / 255.0

    objval = 0.0
    objval_last = 0.0
    k = 1

    for k in range(1, 201):
        tmp = (Y - M) * known
        G = Y + t * (AB - lambda_ * tmp)

        # sigma, u, v = cv2.SVDecomp(G)
        u, sigma, v = np.linalg.svd(G, full_matrices=False)

        sigma = np.maximum(sigma - t, 0.0)
        sigma = np.diag(sigma)
        X = u @ sigma @ v
        t = (1 + np.sqrt(1 + 4 * tlast * tlast)) / 2
        Y = X + (tlast - 1) / t * (X - lastX)

        XX = np.copy(X)
        lastX = np.copy(X)
        tlast = t

        tmp = (X - M) * known
        tr = np.trace(X @ BA)
        objval = traceNorm(X) - tr + lambda_ / 2.0 * np.power(np.linalg.norm(tmp, 'fro'), 2)

        if k >= 2 and -(objval - objval_last) < eps:
            break

        objval_last = objval

    return XX


def TNNR(im0, mask, lower_R, upper_R, lambda_):
    X = im0.astype(np.float64)
    M = np.copy(X)

    W = X.shape[1]
    H = X.shape[0]
    X_rec = np.zeros((H, W), np.float64)
    X_rec_last = np.zeros((H,W), np.float64)
    eps = 0.1

    for R in range(lower_R, upper_R+1):
        for out_iter in range(1, 11):
            u, sigma, v = np.linalg.svd(X, full_matrices=False)
            A = u[:, :R-1]
            A = A.T
            B = v[:R-1, :]

            X_rec = APGL(A, B, X, M, mask, eps, lambda_)

            if out_iter >= 2 and np.linalg.norm(X_rec - X_rec_last, 'fro') / np.linalg.norm(M, 'fro') < 0.01:
                X = np.copy(X_rec)
                break

            X = np.copy(X_rec)
            X_rec_last = np.copy(X_rec)

    return X