import math
import numpy as np
from scipy.linalg import expm


class BM_two_pool:
    def __init__(self, B0, B1, Tsat, Td, offset_b):
        self.B0 = B0
        self.B1 = B1
        self.Tsat = Tsat
        self.Td = Td
        self.offset_b = offset_b
        self.gyro = 42.5764  # MHz/T

    def func(self, dw, T1a, T1b, T2a, T2b, R, M0b):
        w1 = 2 * math.pi * self.gyro * self.B1
        dwa = 2 * math.pi * dw
        dwb = 2 * math.pi * (dw - self.offset_b*self.B0*self.gyro)
        R1a = 1/T1a
        R1b = 1/T1b
        R2a = 1/T2a
        R2b = 1/T2b
        M0a = 1
        kab = R*M0b 
        kba = R*M0a
        M = np.array([0, 0, 1, 0, 0, M0b, 1]) #[7,]
        A = np.array([[-R2a-kab, -dwa, 0, kba, 0, 0, 0],
                      [dwa, -R2a-kab, -w1, 0, kba, 0, 0],
                      [0, w1, -R1a-kab, 0, 0, kba, R1a*M0a],
                      [kab, 0, 0, -R2b-kba, -dwb, 0, 0],
                      [0, kab, 0, dwb, -R2b-kba, -w1, 0],
                      [0, 0, kab, 0, w1, -R1b-kba, R1b*M0b],
                      [0, 0, 0, 0, 0, 0, 0]])
        return np.dot(expm(A*self.Tsat), M) # [7,]

    def func_delay(self, dw, T1a, T1b, T2a, T2b, R, M0b):
        w1 = 2 * math.pi * self.gyro * self.B1
        dwa = 2 * math.pi * dw
        dwb = 2 * math.pi * (dw - self.offset_b*self.B0*self.gyro)
        R1a = 1/T1a
        R1b = 1/T1b
        R2a = 1/T2a
        R2b = 1/T2b
        M0a = 1
        kab = R*M0b 
        kba = R*M0a
        M = np.array([0, 0, 1, 0, 0, M0b, 1])  # (7,)
        A = np.array([[-R2a-kab, -dwa, 0, kba, 0, 0, 0],
                      [dwa, -R2a-kab, -w1, 0, kba, 0, 0],
                      [0, w1, -R1a-kab, 0, 0, kba, R1a*M0a],
                      [kab, 0, 0, -R2b-kba, -dwb, 0, 0],
                      [0, kab, 0, dwb, -R2b-kba, -w1, 0],
                      [0, 0, kab, 0, w1, -R1b-kba, R1b*M0b],
                      [0, 0, 0, 0, 0, 0, 0]])
        Msat = np.dot(expm(A*self.Tsat), M)  # (7,)
        w1 = 0
        Ad = np.array([[-R2a-kab, -dwa, 0, kba, 0, 0, 0],
                      [dwa, -R2a-kab, -w1, 0, kba, 0, 0],
                      [0, w1, -R1a-kab, 0, 0, kba, R1a*M0a],
                      [kab, 0, 0, -R2b-kba, -dwb, 0, 0],
                      [0, kab, 0, dwb, -R2b-kba, -w1, 0],
                      [0, 0, kab, 0, w1, -R1b-kba, R1b*M0b],
                      [0, 0, 0, 0, 0, 0, 0]])
        return np.dot(expm(Ad*self.Td), Msat)  # (7,)

    def forward(self, offset, T1a, T1b, T2a, T2b, R, M0b):
        freq = self.gyro * self.B0 * offset
        Zspec = []
        for i in range(freq.shape[0]):
            Zspec.append(self.func_delay(freq[i], T1a, T1b, T2a, T2b, R, M0b)[2])
        return np.array(Zspec)
    

