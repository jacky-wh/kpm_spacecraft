import numpy as np
from scipy.integrate import odeint
from utils import matrixtoquaternion,getbasis,dynamics,lqr_regulator_k,quaternion2rot,dlqr,lqr_p_k_cal
import matplotlib.pyplot as plt
import os
import trainnet as lka
from data import data_collecter_so3,getbasis
import torch
from scipy.spatial.transform import Rotation as R

#加载模型
suffix = "2-29"
env_names = "Spacecraft_attitude"
method = 'KoopmanU'
root_path = "../Data/" + suffix
for file in os.listdir(root_path):
    if file.startswith(method + "_" + env_names) and file.endswith(".pth"):
        model_path = file

Data_collect = data_collecter_so3(env_names)
udim = Data_collect.udim
Nstates = Data_collect.Nstates

dicts = torch.load(root_path+"/"+model_path,map_location=torch.device('cpu'))

state_dict = dicts["model"]
layer = dicts["layer"]
NKoopman = layer[-1]+39
net = lka.Network(layer,NKoopman,udim)
net.load_state_dict(state_dict)
# device = torch.device("cuda:0")
net.cuda()
net.double()

#升维
def getlift(x):
    x=getbasis(x, n_basis)
    x=x.reshape(1, -1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x= torch.DoubleTensor(x).to(device)
    X_current = net.encode(x)
    X_current = X_current.detach().cpu().numpy().T
    return X_current

def getlift_dnn(x):
    x=x.reshape(1, -1)
    x=x[:,:12]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x= torch.DoubleTensor(x).to(device)
    X_current = net.encode(x)
    X_current = X_current.detach().cpu().numpy().T
    return X_current
#初始参数
dt=0.1
R0=np.array([[0.4830, 0.8365,  0.2588],
             [-0.3209,   0.4441,  -0.8365],
            [-0.8147,  0.3209,  0.4830]])

# R0=np.array([[0.74783335, -0.01635712,  0.66368477],
#              [0.40748675,  0.8005386,  -0.43942187],
#             [-0.52411778,  0.59905713,  0.60533481]])
# R0 = R.random().as_matrix()
wb0 = np.array([[0],[0],[0]])
X0=np.vstack((R0.reshape(-1,1),wb0.reshape(-1,1)))
Rd = np.diag([1,1,1])
Xd=np.vstack((Rd.reshape(-1,1),wb0.reshape(-1,1)))
n_basis=3
Xcurrent=X0

#提取A,B
Alift=net.lA.weight.data.cpu().numpy()
Blift=net.lB.weight.data.cpu().numpy()

#
# Alift=np.load('../Alift.npy')
# Blift=np.load('../Blift.npy')

nx=len(Alift)   #状态维度
nu=len(Blift.T) #状态维度

Q = np.zeros((nx, nx))
Q[:12, :12] = np.eye(12)
R=np.eye(nu)

k=dlqr(np.matrix(Alift),np.matrix(Blift),Q,R)

x0=getlift_dnn(X0)
x_ref_lift=getlift_dnn(Xd)
#对比模拟情况

# x0=getbasis(X0,3)
# x_ref_lift=getbasis(Xd,3)
steps=4000
Turedata = np.zeros((12, steps+1))
Turedata[:,0]=Xcurrent.reshape(-1,)

for i in range(steps):
    # x_ref_lift=getbasis(trajectories[:, i+1],n_basis)
    u = -k@(x0-x_ref_lift)
    u= np.clip(u, -1, 1)
    sn = odeint(dynamics, Xcurrent.reshape(-1,), [0, dt], args=(u,))
    Xcurrent=sn[-1, :]
    Turedata[:, i+1] = Xcurrent
    x0=getlift_dnn(Xcurrent)
    # x0=getbasis(Xcurrent,3)
for i in range(12):
    plt.plot(Turedata[i, :], label="x2",color='red')
    # plt.plot(trajectories[i,:steps])
    plt.axhline(Xd[i])
    plt.show()



