import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BM_two_pool import BM_two_pool


base_dir = r"C:\Users\jwu191\Desktop\BMsim_challenge-main"
df = pd.read_excel(os.path.join(base_dir, "BMsim comparison.xlsx"), 
                   sheet_name="case 2", header=10)
offset = np.concatenate([np.array([-300]), np.arange(-15, 15.1, 0.1)])
Z_MZ = df["M. Zaiss"]
Z_ZZ = df["Z.Zu"]
Z_JH = df["H. Zhang and J. Huang"]
x = np.arange(offset.shape[0])

model = BM_two_pool(B0=3, B1=2, Tsat=2, Td=6.5e-3, offset_b=1.9)
paras = np.array([3, 1.05, 2, 0.1, 50, 5e-4])  # (T1a, T1b, T2a, T2b, R, M0b)
Z_my = model.forward(offset, *paras)
df = pd.DataFrame(Z_my)
df.to_excel(os.path.join(base_dir, "results", "case_2.xlsx"), header=False, index=False)

plt.figure(dpi=300)
plt.plot(x, Z_MZ, label="M. Zaiss")
plt.plot(x, Z_ZZ, label="Z. Zu")
plt.plot(x, Z_JH, label="J. Huang")
plt.plot(x, Z_my, label="My")
plt.legend()
plt.title("Case 2")
plt.show()


