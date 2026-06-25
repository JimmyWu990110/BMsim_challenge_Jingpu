import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BM_two_pool_pulse import BM_two_pool_pulse


base_dir = r"C:\Users\jwu191\Desktop\BMsim_challenge-main"
df = pd.read_excel(os.path.join(base_dir, "BMsim comparison.xlsx"), 
                   sheet_name="case 5", header=10)
offset = np.concatenate([np.array([-300]), np.arange(-2, 2.02, 0.02)])
Z_MZ = df["M. Zaiss (PS)"]
Z_NY = df["Q. Zeng and N. Yadav"]
Z_JH = df["H. Zhang and J. Huang"]
x = np.arange(offset.shape[0])

df_pulse = pd.read_csv(os.path.join(base_dir, "case_5", "rf_pulse.csv"), header=None)
t, amp = np.array(df_pulse[0]), np.array(df_pulse[1])
dt = np.diff(t, append=t[-1] + np.mean(np.diff(t)))  # (200,)

model = BM_two_pool_pulse(B0=3, dt=dt, amp=amp, offset_b=1.9)
paras = np.array([3, 1.05, 2, 0.1, 50, 5e-4])  # (T1a, T1b, T2a, T2b, R, M0b)
Z_my = model.forward(offset, *paras)
df = pd.DataFrame(Z_my)
df.to_excel(os.path.join(base_dir, "results", "case_5.xlsx"), header=False, index=False)

plt.figure(dpi=300)
plt.plot(x, Z_MZ, label="M. Zaiss")
plt.plot(x, Z_NY, label="N. Yadav")
plt.plot(x, Z_JH, label="J. Huang")
plt.plot(x, Z_my, label="My")
plt.legend()
plt.title("Case 5")
plt.show()


