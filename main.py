import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation
from utils import matrixtoquaternion,getbasis,dynamics



def quaternion2rot(quaternion):
    r = Rotation.from_quat(quaternion)
    rot = r.as_matrix()
    return rot
# 初始化参数
dt = 0.01
I=np.diag([0.1,0.2,0.3])
# q=np.array(([0.4618],[0.1917],[0.7999],[0.3320])).reshape(4,)
# R0=quaternion2rot(q)
R0=np.array([[0.74783335, -0.01635712,  0.66368477],
             [0.40748675,  0.8005386,  -0.43942187],
            [-0.52411778,  0.59905713,  0.60533481]])
# R0 = np.array([[0.4830,0.8365,0.2588],
#                [-0.3209, 0.4441, -0.8365],
#                [-0.8147, 0.3209 ,0.4830]])

# R0 = Rotation.random().as_matrix()
wb0 = np.array([[0],[0],[0]])
X0=np.vstack((R0.reshape(-1,1),wb0.reshape(-1,1)))
X0=X0.reshape(-1,)
n_basis=3
Nsim = 200  #Nsim次迭代
Ntraj=400  #Ntraj次轨迹

# 定义函数
# 旋转矩阵变四元数


print('#############获取数据集########################')
start = time.time()
# 产生数据
X = np.empty((12, Ntraj*Nsim))
Y = np.empty((12, Ntraj*Nsim))
U= np.empty((3, Ntraj*Nsim))
for i in range(Ntraj):
    print('第%d轮数据'%i)
    # R0 = Rotation.random().as_matrix()
    # wb0 = np.array([[0], [0], [0]])
    # X0 = np.vstack((R0.reshape(-1, 1), wb0.reshape(-1, 1)))
    # X0 = X0.reshape(-1, )
    Xcurrent=X0
    for j in range(Nsim):
        u = np.random.uniform(-1, 1, (3,))
        sn = odeint(dynamics, Xcurrent, [0, dt], args=(u,))
        X[:,i*Nsim+j]=Xcurrent
        Xcurrent=sn[-1, :]
        Y[:, i * Nsim + j] = sn[-1, :]
        U[:, i * Nsim + j] = u

end = time.time()
print('数据集 time is %d'%(end - start))
np.save('X',X)
np.save('Y',X)
np.save('U',U)


# 构建升维函数
start = time.time()
print('#############数据升维########################')
Xlift =np.empty((12+9*n_basis, Ntraj*Nsim))
Ylift =np.empty((12+9*n_basis, Ntraj*Nsim))
for i in range(Ntraj*Nsim):
    Xlift[:,i]=getbasis(X[:,i],n_basis).reshape(-1,)
    Ylift[:, i] = getbasis(Y[:, i], n_basis).reshape(-1,)

# X[9:12,:]=X[9:12,:]/5
# Y[9:12,:]=Y[9:12,:]/5
# Xlift[9:12,:]=Xlift[9:12,:]/5
# Ylift[9:12,:]=Ylift[9:12,:]/5

end = time.time()
print('数据升维 time is %d'%(end - start))
np.save('Xlift',Xlift)
np.save('Ylift',Ylift)

# 计算
Nlift=len(Xlift)
W=np.vstack((Ylift,X))
V=np.vstack((Xlift,U))
VVt = V@V.T
WVt = W@V.T
M = WVt @np.linalg.inv(VVt)

Alift = M[0:Nlift,0:Nlift]
Blift = M[0:Nlift,Nlift:]
Clift = M[Nlift:,0:Nlift]
np.save('Alift',Alift)
np.save('Blift',Blift)
np.save('Clift',Clift)

# eval 测试画图

Tmax = 1
Nsim = int(Tmax/dt)

# 构建数据保存
xlift=np.empty((Nlift, int(Nsim)))
x_true=np.empty((12, int(Nsim)))
xlift[:,0] = getbasis(X0,n_basis).reshape(-1,)
x_true[:,0] = X0.reshape(-1,)

# Simulate
for i in range(Nsim-1):
    u = np.random.uniform(-2, 2, (3,))
    xlift[:,i+1]=Alift@xlift[:,i]+Blift@u

    #True dynamics
    sn = odeint(dynamics, x_true[:,i], [0, dt], args=(u,))
    x_true[:, i + 1] = sn[-1, :]


# 可视化矩阵参数
# for i in range(12):
#     plt.plot(np.linspace(0, Nsim, Nsim ), x_true[i, :], label="Ture", )
#     plt.scatter(
#         np.linspace(0, Nsim, Nsim ),
#         xlift[i, :],
#         marker="x",
#         c="g",
#         label="koopman",
#     )
#     plt.legend(loc='lower left')
#     plt.title("True_and_koopman", fontsize=15)
#     plt.xlabel("$t$", fontsize=13, x=1)
#     plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
#     plt.show()

# 可视化四元数q
qlift=np.empty((4, int(Nsim)))
q_true=np.empty((4, int(Nsim)))
for i in range(Nsim):
    qlift[:,i]=matrixtoquaternion(xlift[:,i]).reshape(-1,)
    q_true[:,i]=matrixtoquaternion(x_true[:,i]).reshape(-1,)

for i in range(4):
    plt.plot(np.linspace(0, Nsim, Nsim ), q_true[i, :], label="Ture", )
    plt.scatter(
        np.linspace(0, Nsim, Nsim ),
        qlift[i, :],
        marker="x",
        c="g",
        label="koopman",
    )
    plt.legend(loc='lower left')
    plt.title("True_and_koopman", fontsize=15)
    plt.xlabel("$t$", fontsize=13, x=1)
    plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
    plt.show()

# 可视化角速度
for i in range(3):
    plt.plot(np.linspace(0, Nsim, Nsim ), x_true[i+9, :], label="Ture", )
    plt.scatter(
        np.linspace(0, Nsim, Nsim ),
        xlift[i+9, :],
        marker="x",
        c="g",
        label="koopman",
    )
    plt.legend(loc='lower left')
    plt.title("True_and_koopman", fontsize=15)
    plt.xlabel("$t$", fontsize=13, x=1)
    plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
    plt.show()
