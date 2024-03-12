import numpy as np
from scipy.integrate import odeint
from utils import matrixtoquaternion,getbasis,dynamics,lqr_regulator_k,quaternion2rot,dlqr,lqr_p_k_cal
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# R0=np.array([[0.4830, 0.8365,  0.2588],
#              [-0.3209,   0.4441,  -0.8365],
#             [-0.8147,  0.3209,  0.4830]])
R0 = R.random().as_matrix()

dt=0.1
wb0 = np.array([[0],[0],[0]])
X0=np.vstack((R0.reshape(-1,1),wb0.reshape(-1,1)))
n_basis=3
Xcurrent=X0
Alift=np.load('../Alift.npy')
Blift=np.load('../Blift.npy')
Alift_dnn=np.load('../Alift_dnn.npy')
Blift_dnn=np.load('../Blift_dnn.npy')
nx=len(Alift)   #状态维度
# 轨迹
steps=100
trajectories=np.empty((12,steps+1))
trajectories[:,0]=Xcurrent.reshape(-1,)
xkoop=np.empty((nx,steps+1))
xdnn=np.empty((nx,steps+1))
xkoop[:,0]=getbasis(X0,n_basis).reshape(-1,)
xdnn[:,0]=getbasis(X0,n_basis).reshape(-1,)
for i in range(steps):
    u = np.random.uniform(-1, 1, (3,))
    sn = odeint(dynamics, trajectories[:,i], [0, dt], args=(u,))
    trajectories[:, i+1] = sn[-1, :]
    xkoop[:, i + 1] = Alift @ xkoop[:, i] + Blift @ u
    xdnn[:, i + 1] = Alift_dnn @ xdnn[:, i] + Blift_dnn @ u
#
for i in range(12):
    plt.plot(trajectories[i,:steps],color='black')
    plt.plot(xkoop[i, :steps],color='red',marker= 'o')
    plt.plot(xdnn[i, :steps],color='deepskyblue')
    plt.show()


