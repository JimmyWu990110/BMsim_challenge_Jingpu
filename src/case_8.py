import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BM_five_pool_pulse import BM_five_pool_pulse


base_dir = r"C:\Users\jwu191\Desktop\BMsim_challenge-main"
df = pd.read_excel(os.path.join(base_dir, "BMsim comparison.xlsx"), 
                   sheet_name="case 8", header=10)
offset = np.concatenate([np.array([-300]), np.arange(-2, 2.05, 0.05)])
Z_MZ = df["M. Zaiss"]
Z_NY = df["Q. Zeng and N. Yadav"]
Z_JH = df["H. Zhang and J. Huang"]
x = np.arange(offset.shape[0])

amp = 3.7 * 42.5764 * np.concatenate((np.ones(50), np.zeros(1), np.ones(50)))
dt = 1e-4 * np.ones(amp.shape[0])

model = BM_five_pool_pulse(B0=3, dt=dt, amp=amp, offset_b=-3, offset_c=3.5, 
                           offset_d=2, offset_e=-3)
# (T1a, T1b, T1c, T1d, T1e, T2a, T2b, T2c, T2d, T2e, M0b, M0c, M0d, M0e, kba, kca, kda, kea)
paras = np.array([1, 1, 1, 1, 1.3,
                  0.04, 4e-5, 0.1, 0.1, 0.005,
                  0.1351, 0.0009009, 0.0009009, 0.0045,
                  30, 50, 1000, 20])  
Z_my = model.forward(offset, paras)
df = pd.DataFrame(Z_my)
df.to_excel(os.path.join(base_dir, "results", "case_8.xlsx"), header=False, index=False)

# plt.figure(dpi=300)
plt.plot(x, Z_MZ, label="M. Zaiss")
plt.plot(x, Z_NY, label="N. Yadav")
plt.plot(x, Z_JH, label="J. Huang")
plt.plot(x, Z_my, label="My")
plt.legend()
plt.title("Case 8")
plt.show()





