import numpy as np
from scipy.integrate import odeint
from utils import matrixtoquaternion,getbasis,dynamics,lqr_regulator_k,quaternion2rot,dlqr,lqr_p_k_cal
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from data import data_collecter_so3
import os
import torch
import train as lka

#加载模型
suffix = "3-7"
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
NKoopman = layer[-1]+Nstates
net = lka.Network(layer,NKoopman,udim)
net.load_state_dict(state_dict)
# device = torch.device("cuda:0")
net.cuda()
net.double()


def getlift_dnn(x):
    x=x.reshape(1, -1)
    x=x[:,:12]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x= torch.DoubleTensor(x).to(device)
    X_current = net.encode(x)
    X_current = X_current.detach().cpu().numpy().T
    return X_current
R0=np.array([[0.4830, 0.8365,  0.2588],
             [-0.3209,   0.4441,  -0.8365],
            [-0.8147,  0.3209,  0.4830]])
dt=0.1
wb0 = np.array([[0],[0],[0]])
X0=np.vstack((R0.reshape(-1,1),wb0.reshape(-1,1)))

xnn=getlift_dnn(X0)
#提取A,B
Alift_dnn=net.lA.weight.data.cpu().numpy()
Blift_dnn=net.lB.weight.data.cpu().numpy()

nx=len(Alift_dnn)   #状态维度
nu=len(Blift_dnn.T) #状态维度


# 轨迹
steps=100
trajectories=np.empty((12,steps+1))
trajectories[:,0]=X0.reshape(-1,)
xkoop=np.empty((nx,steps+1))
xkoop[:,0]=getlift_dnn(X0).reshape(-1,)
for i in range(steps):
    u = np.random.uniform(-1, 1, (3,))
    sn = odeint(dynamics, trajectories[:,i], [0, dt], args=(u,))
    trajectories[:, i+1] = sn[-1, :]
    xkoop[:, i + 1] = Alift_dnn @ xkoop[:, i] + Blift_dnn @ u
#
for i in range(12):
    plt.plot(trajectories[i,:steps],color='black')
    plt.plot(xkoop[i, :steps],color='red',marker= 'o')
    plt.show()


