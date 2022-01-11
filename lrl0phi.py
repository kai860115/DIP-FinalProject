import cv2
import numpy as np

from utils import norm_tv, PSNR
from tnnr import TNNR
from common import g

class LRL0PHI:
    def __init__(self, I, mask, u0, rho, dt, lambda_l0, lambda_rank, k):
        self.I = I.astype(np.float64)
        self.mask = mask.astype(np.float64) / 255.0

        self.H, self.W = self.I.shape

        self.U = np.zeros((self.H, self.W), np.float64)
        self.U_last = np.zeros((self.H, self.W), np.float64)
        self.M = u0.astype(np.float64)
        self.Y = np.zeros((self.H, self.W), dtype=np.float64)

        self.kernelx_plus = np.array([[0.0, -1.0, 1.0]])
        self.kernelx_minus = np.array([[-1.0, 1.0, 0.0]])
        self.kernely_plus = np.array([[0.0], [-1.0], [1.0]])
        self.kernely_minus = np.array([[-1.0], [1.0], [0.0]])

        self.rho = rho
        self.dt = dt
        self.alpha = 1.0
        self.lambda_rank = lambda_rank
        self.lambda_l0 = lambda_l0
        self.k = k

    def sub_1(self, K):
        epsilon = 1e-4
        I = {}
        rows = self.U.shape[0]
        cols = self.U.shape[1]
        for i in range(rows):
            for j in range(cols):
                index = i * cols + j
                temp = {}
                if 'G' not in temp:
                    temp['G'] = []
                temp['G'] += [int(index)]
                temp['w'] = 1
                temp['w_mask'] = 0 if self.mask[i][j] == 0 else 1
                temp['I_mean'] = temp['w_mask'] * self.I[i][j]
                
                temp['M_mean'] = self.M[i][j]
                temp['Y_mean'] = self.Y[i][j]
                temp['Y'] = (temp['w'] * self.rho * (temp['M_mean'] - temp['Y_mean']) + 2 * temp['w_mask'] * temp['I_mean']) / (temp['w'] * self.rho + 2 * temp['w_mask'])
                
                if 'N' not in temp:
                    temp['N'] = []
                if 'c' not in temp:
                    temp['c'] = {}
                if i != 0:
                    temp['N'].append((i - 1) * cols + j)
                    temp['c'][(i - 1) * cols + j] = 1
                if i != rows - 1:
                    temp['N'].append((i + 1) * cols + j)
                    temp['c'][(i + 1) * cols + j] = 1
                if j != 0:
                    temp['N'].append(i * cols + j - 1)
                    temp['c'][i * cols + j - 1] = 1
                if j != cols - 1:
                    temp['N'].append(i * cols + j + 1)
                    temp['c'][i * cols + j + 1] = 1
                I[index] = temp

        for i, temp in I.items():
            for j in temp['G']:
                self.U[j // cols][j % cols] = temp['Y']
        
        beta = 0
        iteration = 0

        while True:
            I_copy = I.copy()
            it = iter(I_copy.items())
            try:
                while True:
                    i, ele = next(it)
                    while i not in I:
                        i, ele = next(it)
                    j_idx = 0
                    while j_idx < len(I[i]['N']):
                        j = I[i]['N'][j_idx]
                        temp1 = I[i]
                        temp2 = I[j]
                        value1 = (temp1['w'] * temp1['w_mask'] * self.rho * np.power(temp1['I_mean'] - temp1['M_mean'] + temp1['Y_mean'], 2) / (temp1['w'] * self.rho + 2 * temp1['w_mask'])) + (temp2['w'] * temp2['w_mask'] * self.rho * np.power(temp2['I_mean'] - temp2['M_mean'] + temp2['Y_mean'], 2) / (temp2['w'] * self.rho + 2 * temp2['w_mask'])) + (beta * temp1['c'][j])

                        X = (temp1['w'] * self.rho * (temp1['M_mean'] - temp1['Y_mean']) + 2 * temp1['w_mask'] * temp1['I_mean'] + temp2['w'] * self.rho * (temp2['M_mean'] - temp2['Y_mean']) + 2 * temp2['w_mask'] * temp2['I_mean']) / (temp1['w'] * self.rho + 2 * temp1['w_mask'] + temp2['w'] * self.rho + 2 * temp2['w_mask'])
                        value2 = (temp1['w'] * self.rho / 2.0 * np.power(X - temp1['M_mean'] + temp1['Y_mean'], 2)) + (temp1['w_mask'] * np.power(X - temp1['I_mean'], 2)) + (temp2['w'] * self.rho / 2.0 * np.power(X - temp2['M_mean'] + temp2['Y_mean'], 2)) + (temp2['w_mask'] * np.power(X - temp2['I_mean'], 2))

                        if temp2['Y'] < temp1['Y']:
                            Xi = (temp1['w'] * self.rho * (temp1['M_mean'] - temp1['Y_mean']) + 2 * temp1['w_mask'] * temp1['I_mean'] + temp2['w'] * self.rho * (temp2['M_mean'] - temp2['Y_mean'] + 1.0) + 2 * temp2['w_mask'] * (temp2['I_mean'] + 1.0)) / (temp1['w'] * self.rho + 2 * temp1['w_mask'] + temp2['w'] * self.rho + 2 * temp2['w_mask'])
                            Xj = Xi - 1.0
                        else:
                            Xi = (temp1['w'] * self.rho * (temp1['M_mean'] - temp1['Y_mean']) + 2 * temp1['w_mask'] * temp1['I_mean'] + temp2['w'] * self.rho * (temp2['M_mean'] - temp2['Y_mean'] - 1.0) + 2 * temp2['w_mask'] * (temp2['I_mean'] - 1.0)) / (temp1['w'] * self.rho + 2 * temp1['w_mask'] + temp2['w'] * self.rho + 2 * temp2['w_mask'])
                            Xj = Xi + 1.0
                            
                        value3 = (temp1['w'] * self.rho / 2.0 * np.power(Xi - temp1['M_mean'] + temp1['Y_mean'], 2)) + (temp1['w_mask'] * np.power(Xi - temp1['I_mean'], 2)) + (temp2['w'] * self.rho / 2.0 * np.power(Xj - temp2['M_mean'] + temp2['Y_mean'], 2)) + (temp2['w_mask'] * np.power(Xj - temp2['I_mean'], 2)) + self.k * self.lambda_l0
                        value1 = 0 if value1 <= epsilon else np.float64(value1)
                        value2 = 0 if value2 <= epsilon else np.float64(value2)
                        value3 = 0 if value3 <= epsilon else np.float64(value3)

                        if value3 < min(value1, value2):
                            temp1['Y'] = Xi
                            temp2['Y'] = Xj
                            j_idx += 1
                        elif value2 <= min(value1, value3):
                            temp_value = j
                            
                            I[i]['Y'] = X
                            I[i]['Y_mean'] = (I[i]['Y_mean'] * I[i]['w'] + I[j]['Y_mean'] * I[j]['w']) / (I[i]['w'] + I[j]['w'])
                            I[i]['M_mean'] = (I[i]['M_mean'] * I[i]['w'] + I[j]['M_mean'] * I[j]['w']) / (I[i]['w'] + I[j]['w'])
                            if I[i]['w_mask'] == 0 and I[j]['w_mask'] == 0:
                                I[i]['I_mean'] = 0
                            else:
                                I[i]['I_mean'] = (I[i]['I_mean'] * I[i]['w_mask'] + I[j]['I_mean'] * I[j]['w_mask']) / (I[i]['w_mask'] + I[j]['w_mask'])
                            I[i]['w'] = I[i]['w'] + I[j]['w']
                            I[i]['w_mask'] = I[i]['w_mask'] + I[j]['w_mask']
                            I[i]['G'] += I[j]['G']
                            I[i]['G'].sort()
                            I[i]['c'].pop(temp_value)
                            I[i]['N'].remove(j)

                            for k in I[temp_value]['N']:
                                if k == i:
                                    continue
                                if k in I[i]['N']:
                                    I[i]['c'][k] += I[temp_value]['c'][k]
                                    I[k]['c'][i] = I[i]['c'][k]
                                else:
                                    I[i]['N'].append(k)
                                    I[k]['N'].append(i)
                                    I[i]['c'][k] = I[temp_value]['c'][k]
                                    I[k]['c'][i] = I[temp_value]['c'][k]
                                
                                I[k]['c'].pop(temp_value)
                                I[k]['N'].remove(temp_value)

                            I.pop(temp_value)
                            i, ele = next(it)
                            while i not in I:
                                i, ele = next(it)
                            break

                        else:
                            j_idx += 1
                    
            except StopIteration:
                pass

            iteration += 1
            beta = g(iteration, K, self.lambda_l0)
            if beta > self.lambda_l0:
                break
        
        for i, ele in I.items():
            for j in ele['G']:
                self.U[j // cols][j % cols] = ele['Y']


    def sub_2(self):
        A = self.U + self.Y
        mask = 255 * np.ones((self.H, self.W), np.uint8)
        lambda_ = self.rho / 2.0 / self.lambda_rank / self.alpha

        A = np.maximum(A, 0)
        A = np.minimum(A, 255)
        At = A.astype(np.uint8)
        self.M = TNNR(At, mask, 9, 9, lambda_)

    def sub_3(self):
        self.Y = self.Y + self.U - self.M

    def compute(self, K, max_iter, path, original):
        for iter_ in range(max_iter):
            print("=====================")
            print(f"Iter = {iter_ + 1}")
            self.sub_1(K)
            print("L0 done")
            self.sub_2()
            print("LR done")
            self.sub_3()

            output = self.U.astype(np.uint8)
            cv2.imwrite(f"{path}/lrl0phi_{iter_ + 1}.png", output)
            mask = 255.0 * self.mask
            psnr = PSNR(original, self.U, mask)

            filePath = f"{path}/{iter_ + 1}.txt"
            with open(filePath, 'w') as f:
                f.write(str(psnr))
            print(f"PSNR = {psnr}")
            print(f"result psnr in {filePath}")

            if iter_ >= 1 and cv2.norm(self.U, self.U_last) / (cv2.norm(self.U_last) + 1e-12) < 1e-3:
                break
            print(f"relative error = {cv2.norm(self.U, self.U_last) / (cv2.norm(self.U_last) + 1e-12)}")
            self.U_last = np.copy(self.U)
            print()

        return self.U

            