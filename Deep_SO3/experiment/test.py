
import numpy as np
from scipy.integrate import odeint
from utils import matrixtoquaternion,getbasis,dynamics,lqr_regulator_k,quaternion2rot,dlqr,lqr_p_k_cal
import matplotlib.pyplot as plt
import os
import train as lka
from data import data_collecter_so3
import torch
from scipy.spatial.transform import Rotation
import random
from train import Klinear_loss
import torch.nn as nn


def getlift(x):
    x=x.reshape(1, -1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x= torch.DoubleTensor(x).to(device)
    X_current = net.encode(x)
    X_current = X_current.detach().cpu().numpy().T
    return X_current

def netforwad(x,u):
    x=x.reshape(1, -1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x= torch.DoubleTensor(x).to(device)
    u= torch.DoubleTensor(u).to(device)
    X_current = net.encode(x)
    X_next = net.forward(X_current, u).detach().cpu().numpy().T
    # X_current = X_current.detach().cpu().numpy().T
    return X_next



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

test_data_path="../Data/dataset/test/{}.npy".format(env_names)
Ktest_data = np.load("../Data/dataset/test/{}.npy".format(env_names))
print("test data ok!,shape:", Ktest_data.shape)
Ktest_data2=Ktest_data[:,0:1,:]
Ktest_data1=Ktest_data[:,0,:]
print(Ktest_data1.shape)
u_dim=3
gamma=0.99
Nstate=12
mse_loss = nn.MSELoss()

#可视化
steps=300
trajectories=np.empty((12,steps+1))
trajectories[:,0]=Ktest_data1[0,3:].reshape(-1,)
dt=0.1

#提取A,B
Alift=net.lA.weight.data.cpu().numpy()
Blift=net.lB.weight.data.cpu().numpy()
nx=len(Alift)   #状态维度
nu=len(Blift.T) #状态维度
xkoop=np.empty((nx,steps+1))
x01=getlift(trajectories[:,0])
xkoop[:,0]=x01.reshape(-1,)

xkoop1=np.empty((nx,steps+1))
x01=getlift(trajectories[:,0])
xkoop1[:,0]=x01.reshape(-1,)
for i in range(steps):
    u =Ktest_data1[i,:3]
    sn = odeint(dynamics, trajectories[:,i], [0, dt], args=(u,))
    trajectories[:, i+1] = sn[-1, :]
    xkoop1[:, i + 1] = Alift @ xkoop1[:, i] + Blift @ u
    xkoop[:, i + 1] = netforwad(trajectories[:,i],u).reshape(-1,)
#
for i in range(12):
    plt.plot(trajectories[i,:steps],color='red')
    plt.plot(xkoop[i, :steps],color='blue')
    plt.plot(xkoop1[i, :steps])
    plt.show()

with torch.no_grad():
    Kloss= Klinear_loss(Ktest_data2, net, mse_loss, u_dim, gamma, Nstate, all_loss=0)
    loss = Kloss
    print(loss)
