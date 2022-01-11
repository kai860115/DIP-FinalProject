import cv2
import numpy as np

from utils import norm_tv, PSNR
from tnnr import TNNR

class LRTV:
    def __init__(self, I, mask, u0, rho, dt, lambda_tv, lambda_rank):
        self.I = I.astype(np.float64)
        self.W = I.shape[1]
        self.H = I.shape[0]
        self.mask = mask.astype(np.float64) / 255
        self.U = u0.astype(np.float64)
        self.U_last = np.zeros((self.H, self.W), np.float64)
        self.M = np.zeros((self.H, self.W), dtype=np.float64)
        self.Y = np.zeros((self.H, self.W), dtype=np.float64)
        self.kernelx_plus = np.array([[0.0, -1.0, 1.0]])
        self.kernelx_minus = np.array([[-1.0, 1.0, 0.0]])
        self.kernely_plus = np.array([[0.0], [-1.0], [1.0]])
        self.kernely_minus = np.array([[-1.0], [1.0], [0.0]])
        self.rho = rho
        self.dt = dt
        self.alpha = 1.0
        self.lambda_tv = lambda_tv
        self.lambda_rank = lambda_rank

    def getU(self):
        return self.U

    def getM(self):
        return self.M

    def getY(self):
        return self.Y

    def sub_1_val(self, X):
        val = 0.0
        part1 = X - self.I
        part1 = self.mask * part1
        part1 = np.power(part1, 2.0)

        val += part1.sum()
        val += self.lambda_tv * norm_tv(X)

        part2 = X - self.M + self.Y
        part2 = np.power(part2, 2.0)
        part2 = self.rho / 2.0 * part2

        val += part2.sum()
        return val

    def sub1(self):
        epsilon = 1e-4
        totalIterations = 3000
        iter = 0

        while (iter < totalIterations):
            part1 = -2.0 * (self.U - self.I)
            part1 = part1 * self.mask
            part1 = part1 - self.rho * (self.U - self.M + self.Y)

            grad3 = self.rho * (self.U - self.M + self.Y)

            grad_x_plus = cv2.filter2D(self.U, -1, self.kernelx_plus)
            grad_x_minus = cv2.filter2D(self.U, -1, self.kernelx_minus)
            grad_y_plus = cv2.filter2D(self.U, -1, self.kernely_plus)
            grad_y_minus = cv2.filter2D(self.U, -1, self.kernely_minus)

            U_xx = cv2.filter2D(grad_x_minus, -1, self.kernelx_plus)
            U_yy = cv2.filter2D(grad_y_minus, -1, self.kernely_plus)

            U_x = (grad_x_minus + grad_x_plus) / 2.0
            U_y = (grad_y_minus + grad_y_plus) / 2.0

            U_y_x_plus = cv2.filter2D(U_y, -1, self.kernelx_plus)
            U_y_x_minus = cv2.filter2D(U_y, -1, self.kernelx_minus)
            U_xy = (U_y_x_plus + U_y_x_minus) / 2.0

            uy_uy = U_y * U_y + epsilon
            ux_ux = U_x * U_x + epsilon
            ux_uy = U_x * U_y

            uxx_uy_uy = U_xx * uy_uy
            uyy_ux_ux = U_yy * ux_ux
            uxy_ux_uy = U_xy * ux_uy
            numerator = uxx_uy_uy - 2.0 * uxy_ux_uy + uyy_ux_ux

            denominator = ux_ux + uy_uy - epsilon
            denominator = np.power(denominator, 1.5)

            part2 = numerator / denominator
            part2 = self.lambda_tv * part2

            step_size = self.dt
            tau = 0.8
            c = 0.25

            f_0 = self.sub_1_val(self.U)

            while True:
                U_new = self.U + step_size * (part1 + part2)

                f_1 = self.sub_1_val(U_new)
                m = (part1 + part2) * (part1 + part2)
                m_ = m.sum() / self.W / self.H
                t = m_ * c

                if f_0 - f_1 > step_size * t:
                    break
                step_size *= tau
                if step_size < 1e-15:
                    print("Ooooooops! Fail to find descending direction")
                    return

            U_1 = self.U + step_size * (part1 + part2)
            iter += 1

            TVlast = norm_tv(self.U)
            TV = norm_tv(U_1)
            objval = self.sub_1_val(U_1)
            objval_last = self.sub_1_val(self.U)

            print(objval)
            print(f"error = {abs(objval - objval_last) / objval_last}")
            if (objval - objval_last) > 0 :
                print('ascending')
            else:
                print('descending')
            
            if iter >= 20 and abs(objval - objval_last) / objval_last < 1e-3:
                print(f'TV = ({TVlast}) -> {TV}')
                print(f'objval = ({objval_last}) -> {objval}')
                break

            self.U = U_1
            objval_last = objval
        

    def sub2(self):
        A = self.U + self.Y
        mask = 255 * np.ones((self.H, self.W), dtype=np.uint8)
        lambda_ = self.rho / 2.0 / self.lambda_rank / self.alpha

        A = np.maximum(A, 0)
        A = np.minimum(A, 255)
        At = A.astype(np.uint8)
        self.M = TNNR(At, mask, 9, 9, lambda_)

    def sub3(self):
        self.Y = self.Y + self.U - self.M

    def compute(self):
        max_iter = 30
        for iter in range(max_iter):
            print("=====================")
            print(f"Iter = {iter + 1}")

            self.sub1()
            self.sub2()
            self.sub3()

            if iter >= 5 and cv2.norm(self.U, self.U_last) < 1.5e-3:
                break
            
            print(f"relative error = {1.0 * cv2.norm(self.U, self.U_last) / (cv2.norm(self.U_last)+1e-20)}")
            self.U_last = np.copy(self.U)

            print(f'PSNR = {PSNR(self.I, self.U, self.mask)}')

        return self.U
