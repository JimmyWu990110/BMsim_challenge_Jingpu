import math
import numpy as np
from scipy.linalg import expm


class BM_two_pool_pulse:
    def __init__(self, B0, dt, amp, offset_b):
        self.B0 = B0
        self.dt = dt  # (N,)
        self.dw1 = 2 * math.pi * amp  # (N,) hz -> rad/s
        self.offset_b = offset_b
        self.gyro = 42.5764  # MHz/T

    def make_A(self, dw, T1a, T1b, T2a, T2b, R, M0b, i):
        w1 = self.dw1[i]
        dwa = 2 * math.pi * dw
        dwb = 2 * math.pi * (dw - self.offset_b*self.B0*self.gyro)
        R1a = 1/T1a
        R1b = 1/T1b
        R2a = 1/T2a
        R2b = 1/T2b
        M0a = 1
        kab = R*M0b 
        kba = R*M0a
        A = np.array([[-R2a-kab, -dwa, 0, kba, 0, 0, 0],
                      [dwa, -R2a-kab, -w1, 0, kba, 0, 0],
                      [0, w1, -R1a-kab, 0, 0, kba, R1a*M0a],
                      [kab, 0, 0, -R2b-kba, -dwb, 0, 0],
                      [0, kab, 0, dwb, -R2b-kba, -w1, 0],
                      [0, 0, kab, 0, w1, -R1b-kba, R1b*M0b],
                      [0, 0, 0, 0, 0, 0, 0]])
        return A

    def simulate_one_offset(self, dw, T1a, T1b, T2a, T2b, R, M0b):
        M = np.array([0, 0, 1, 0, 0, M0b, 1])
        for i in range(len(self.dw1)):
            A = self.make_A(dw, T1a, T1b, T2a, T2b, R, M0b, i)
            M = expm(A * self.dt[i]) @ M
        return M

    def forward(self, offset, T1a, T1b, T2a, T2b, R, M0b):
        freq = self.gyro * self.B0 * offset
        Zspec = []
        for dw in freq:
            Zspec.append(self.simulate_one_offset(dw, T1a, T1b, T2a, T2b, R, M0b)[2])
        return np.array(Zspec)
    

