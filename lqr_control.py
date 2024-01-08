import numpy as np
from scipy.integrate import odeint
from utils import matrixtoquaternion,hat,getbasis,dynamics,dm_to_array,lqr_regulator_k
import matplotlib.pyplot as plt
from test import dlqr

# def quaternion2rot(quaternion):
#     r = Rotation.from_quat(quaternion)
#     rot = r.as_matrix()
#     return rot

R0=np.array([[0.74783335, -0.01635712,  0.66368477],
             [0.40748675,  0.8005386,  -0.43942187],
            [-0.52411778,  0.59905713,  0.60533481]])
wb0 = np.array([[0],[0],[0]])
X0=np.vstack((R0.reshape(-1,1),wb0.reshape(-1,1)))

Rd = np.diag([1,1,1])
Xd=np.vstack((Rd.reshape(-1,1),wb0.reshape(-1,1)))
n_basis=3
Xcurrent=X0

Alift=np.load('Alift.npy')
Blift=np.load('Blift.npy')

##  初始化变量
nx=len(Alift)   #状态维度
nu=len(Blift.T) #状态维度

Q = np.zeros((nx, nx))
Q[:12, :12] = np.eye(12)*10
# Q = np.eye(nx)
R=np.eye(nu)
Kopt = lqr_regulator_k(np.matrix(Alift),np.matrix(Blift),Q,R)
print(Kopt)
k=dlqr(np.matrix(Alift),np.matrix(Blift),Q,R)
print(k)
# 轨迹
steps=1000
trajectories=np.empty((12,steps+1))
trajectories[:,0]=Xcurrent.reshape(-1,)
xkoop=np.empty((nx,steps+1))
xkoop[:,0]=getbasis(X0,n_basis).reshape(-1,)
for i in range(steps):
    u = np.random.uniform(-1, 1, (3,))*0.1
    sn = odeint(dynamics, trajectories[:,i], [0, 0.01], args=(u,))
    trajectories[:, i+1] = sn[-1, :]
    xkoop[:, i + 1] = Alift @ xkoop[:, i] + Blift @ u

# for i in range(12):
#     plt.plot(trajectories[i,:steps])
#     plt.plot(xkoop[i, :steps])
#     plt.show()



x0=getbasis(X0,n_basis)

Turedata = np.zeros((12, steps+1))
Turedata[:,0]=Xcurrent.reshape(-1,)

x_ref_lift=getbasis(Xd,n_basis)
for i in range(steps):
    # x_ref_lift=getbasis(trajectories[:, i+1],n_basis)
    u = -k@(x0-x_ref_lift)
    u= np.clip(u, -1, 1)
    sn = odeint(dynamics, Xcurrent.reshape(-1,), [0, 0.01], args=(u,))
    Xcurrent=sn[-1, :]
    Turedata[:, i+1] = Xcurrent
    x0=getbasis(Xcurrent,n_basis)

# for i in range(12):
#     plt.plot(Turedata[i, :], label="x2",color='red')
#     # plt.plot(trajectories[i,:steps])
#     plt.axhline(Xd[i])
#     plt.show()

# 可视化四元数q
qlift=np.empty((4, int(steps)))
q_true=np.empty((4, int(steps)))
for i in range(steps):
    qlift[:,i]=matrixtoquaternion(Turedata[:,i]).reshape(-1,)
    # q_true[:,i]=matrixtoquaternion(trajectories[:,i]).reshape(-1,)
    q_true[:, i] = matrixtoquaternion(Xd).reshape(-1, )

for i in range(4):
    plt.plot(np.linspace(0, steps, steps ), qlift[i, :], label="Ture",color='red'  )
    plt.plot(np.linspace(0, steps, steps ), q_true[i, :], label="Ture" )
    # plt.scatter(
    #     np.linspace(0, steps, steps ),
    #     qlift[i, :],
    #     marker="x",
    #     c="g",
    #     label="koopman",
    # )
    plt.legend(loc='lower left')
    plt.title("True_and_koopman", fontsize=15)
    plt.xlabel("$t$", fontsize=13, x=1)
    plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
    plt.show()

for i in range(3):
    plt.plot(Turedata[i+9, :], label="x2",color='red')
    # plt.plot(trajectories[i,:steps])
    plt.axhline(Xd[i+9])
    plt.show()