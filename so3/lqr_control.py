import numpy as np
from scipy.integrate import odeint
from utils import matrixtoquaternion,getbasis,dynamics,lqr_regulator_k,quaternion2rot,dlqr,lqr_p_k_cal
import matplotlib.pyplot as plt


R0=np.array([[0.4830, 0.8365,  0.2588],
             [-0.3209,   0.4441,  -0.8365],
            [-0.8147,  0.3209,  0.4830]])


wb0 = np.array([[0],[0],[0]])
X0=np.vstack((R0.reshape(-1,1),wb0.reshape(-1,1)))
dt=0.1
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
Q[:12, :12] = np.eye(12) * 10
# Q[:9, :9] = np.eye(9)*100
# Q[9:12, 9:12] = np.eye(3)
# Q = np.eye(nx)* 10
R=np.eye(nu)
Kopt = lqr_regulator_k(np.matrix(Alift),np.matrix(Blift),Q,R)

k=dlqr(np.matrix(Alift),np.matrix(Blift),Q,R)

# 轨迹
steps=1000
trajectories=np.empty((12,steps+1))
trajectories[:,0]=Xcurrent.reshape(-1,)
xkoop=np.empty((nx,steps+1))
xkoop[:,0]=getbasis(X0,n_basis).reshape(-1,)
for i in range(steps):
    u = np.random.uniform(-1, 1, (3,))*0.1
    sn = odeint(dynamics, trajectories[:,i], [0, dt], args=(u,))
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
# p, k1 =lqr_p_k_cal(np.matrix(Alift), np.matrix(Blift),np.matrix(Q),np.matrix(R))
for i in range(steps):
    # x_ref_lift=getbasis(trajectories[:, i+1],n_basis)
    u = -k@(x0-x_ref_lift)
    u= np.clip(u, -1, 1)
    sn = odeint(dynamics, Xcurrent.reshape(-1,), [0, dt], args=(u,))
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