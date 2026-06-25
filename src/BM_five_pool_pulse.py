import math
import numpy as np
from scipy.linalg import expm


class BM_five_pool_pulse:
    def __init__(self, B0, dt, amp, offset_b, offset_c, offset_d, offset_e):
        self.B0 = B0
        self.dt = dt  # (N,)
        self.dw1 = 2 * math.pi * amp  # (N,) hz -> rad/s
        self.offset_b = offset_b
        self.offset_c = offset_c
        self.offset_d = offset_d
        self.offset_e = offset_e
        self.gyro = 42.5764  # MHz/T

    def make_A(self, dw, T1a, T1b, T1c, T1d, T1e, T2a, T2b, T2c, T2d, T2e, 
               M0b, M0c, M0d, M0e, kba, kca, kda, kea, i):
        w1 = self.dw1[i]
        dwa = 2 * math.pi * dw
        dwb = 2 * math.pi * (dw - self.offset_b*self.B0*self.gyro)
        dwc = 2 * math.pi * (dw - self.offset_c*self.B0*self.gyro)
        dwd = 2 * math.pi * (dw - self.offset_d*self.B0*self.gyro)
        dwe = 2 * math.pi * (dw - self.offset_e*self.B0*self.gyro)
        R1a = 1/T1a
        R1b = 1/T1b
        R1c = 1/T1c
        R1d = 1/T1d
        R1e = 1/T1e
        R2a = 1/T2a
        R2b = 1/T2b
        R2c = 1/T2c
        R2d = 1/T2d
        R2e = 1/T2e
        M0a = 1
        kab = kba * (M0b/M0a)
        kac = kca * (M0c/M0a)
        kad = kda * (M0d/M0a)
        kae = kea * (M0e/M0a)
        ka = kab+kac+kad+kae
        A = np.array([[-R2a-ka, -dwa, 0, kba, 0, 0, kca, 0, 0, kda, 0, 0, kea, 0, 0, 0],
                      [dwa, -R2a-ka, -w1, 0, kba, 0, 0, kca, 0, 0, kda, 0, 0, kea, 0, 0],
                      [0, w1, -R1a-ka, 0, 0, kba, 0, 0, kca, 0, 0, kda, 0, 0, kea, R1a*M0a],
                      [kab, 0, 0, -R2b-kba, -dwb, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, kab, 0, dwb, -R2b-kba, -w1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, kab, 0, w1, -R1b-kba, 0, 0, 0, 0, 0, 0, 0, 0, 0, R1b*M0b],
                      [kac, 0, 0, 0, 0, 0, -R2c-kca, -dwc, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, kac, 0, 0, 0, 0, dwc, -R2c-kca, -w1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, kac, 0, 0, 0, 0, w1, -R1c-kca, 0, 0, 0, 0, 0, 0, R1c*M0c],
                      [kad, 0, 0, 0, 0, 0, 0, 0, 0, -R2d-kda, -dwd, 0, 0, 0, 0, 0],
                      [0, kad, 0, 0, 0, 0, 0, 0, 0, dwd, -R2d-kda, -w1, 0, 0, 0, 0],
                      [0, 0, kad, 0, 0, 0, 0, 0, 0, 0, w1, -R1d-kda, 0, 0, 0, R1d*M0d],
                      [kae, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -R2e-kea, -dwe, 0, 0],
                      [0, kae, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dwe, -R2e-kea, -w1, 0],
                      [0, 0, kae, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1, -R1e-kea, R1e*M0e],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        return A

    def simulate_one_offset(self, dw, paras):
        M0b, M0c, M0d, M0e = paras[10], paras[11], paras[12], paras[13]
        M = np.array([0, 0, 1, 0, 0, M0b, 0, 0, M0c, 0, 0, M0d, 0, 0, M0e, 1])
        for i in range(len(self.dw1)):
            A = self.make_A(dw, *paras, i)
            M = expm(A * self.dt[i]) @ M
        return M

    def forward(self, offset, paras):
        freq = self.gyro * self.B0 * offset
        Zspec = []
        for dw in freq:
            Zspec.append(self.simulate_one_offset(dw, paras)[2])
        return np.array(Zspec)
    
