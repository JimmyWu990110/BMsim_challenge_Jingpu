import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BM_five_pool_pulse import BM_five_pool_pulse


base_dir = r"C:\Users\jwu191\Desktop\BMsim_challenge-main"
df = pd.read_excel(os.path.join(base_dir, "BMsim comparison.xlsx"), 
                   sheet_name="case 7", header=10)
offset = np.concatenate([np.array([-300]), np.arange(-15, 15.1, 0.1)])
Z_MZ = df["M. Zaiss"]
Z_NY = df["Q. Zeng and N. Yadav"]
Z_JH = df["H. Zhang and J. Huang"]
x = np.arange(offset.shape[0])

df_pulse = pd.read_csv(os.path.join(base_dir, "case_7", "rf_pulse.csv"), header=None)
t, amp = np.array(df_pulse[0]), np.array(df_pulse[1])
dt = np.diff(t, append=t[-1] + np.mean(np.diff(t)))  # (200,)
amp_full = amp.copy()
for i in range(35):
    amp_full = np.concatenate((amp_full, np.zeros(20)))
    amp_full = np.concatenate((amp_full, amp))
dt = dt[0] * np.ones((amp_full.shape[0]))

model = BM_five_pool_pulse(B0=3, dt=dt, amp=amp_full, offset_b=-3, offset_c=3.5, 
                           offset_d=2, offset_e=-3)
# (T1a, T1b, T1c, T1d, T1e, T2a, T2b, T2c, T2d, T2e, M0b, M0c, M0d, M0e, kba, kca, kda, kea)
paras = np.array([1, 1, 1, 1, 1.3,
                  0.04, 4e-5, 0.1, 0.1, 0.005,
                  0.1351, 0.0009009, 0.0009009, 0.0045,
                  30, 50, 1000, 20])  
Z_my = model.forward(offset, paras)
df = pd.DataFrame(Z_my)
df.to_excel(os.path.join(base_dir, "results", "case_7.xlsx"), header=False, index=False)

plt.figure(dpi=300)
plt.plot(x, Z_MZ, label="M. Zaiss")
plt.plot(x, Z_NY, label="N. Yadav")
plt.plot(x, Z_JH, label="J. Huang")
plt.plot(x, Z_my, label="My")
plt.legend()
plt.title("Case 7")
plt.show()





