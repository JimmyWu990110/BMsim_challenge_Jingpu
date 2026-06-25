import math
import numpy as np
from scipy.linalg import expm


class BM_five_pool:
    def __init__(self, B0, B1, Tsat, Td, offset_b, offset_c, offset_d, offset_e):
        self.B0 = B0
        self.B1 = B1
        self.Tsat = Tsat
        self.Td = Td
        self.offset_b = offset_b
        self.offset_c = offset_c
        self.offset_d = offset_d
        self.offset_e = offset_e
        self.gyro = 42.5764  # MHz/T

    def func(self, dw, T1a, T1b, T1c, T1d, T1e, T2a, T2b, T2c, T2d, T2e, 
             M0b, M0c, M0d, M0e, kba, kca, kda, kea):
        w1 = 2 * math.pi * self.gyro * self.B1
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
        M = np.array([0, 0, 1, 0, 0, M0b, 0, 0, M0c, 0, 0, M0d, 0, 0, M0e, 1]) #[16,]
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
        return np.dot(expm(A*self.Tsat), M)  # (16,)

    def func_delay(self, dw, T1a, T1b, T1c, T1d, T1e, T2a, T2b, T2c, T2d, T2e, 
                   M0b, M0c, M0d, M0e, kba, kca, kda, kea):
        w1 = 2 * math.pi * self.gyro * self.B1
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
        M = np.array([0, 0, 1, 0, 0, M0b, 0, 0, M0c, 0, 0, M0d, 0, 0, M0e, 1]) #[16,]
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
        Msat = np.dot(expm(A*self.Tsat), M)  # (16,)
        w1 = 0
        Ad = np.array([[-R2a-ka, -dwa, 0, kba, 0, 0, kca, 0, 0, kda, 0, 0, kea, 0, 0, 0],
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
        return np.dot(expm(Ad*self.Td), Msat)  # (16,)

    def forward(self, offset, paras):
        freq = self.gyro * self.B0 * offset
        Zspec = []
        for i in range(freq.shape[0]):
            Zspec.append(self.func_delay(freq[i], *paras)[2])
        return np.array(Zspec)

    
