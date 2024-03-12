import numpy as np
import gym
import random
from scipy.integrate import odeint
import scipy.linalg
from copy import copy
from gym import spaces
import sys
from scipy.spatial.transform import Rotation as R



class Spacecraft_so3():
    def __init__(self) -> None:
        self.I=np.asarray([[3, 0, 0], [0, 4, 0], [0, 0, 5]])
        self.dt = 0.1
        self.Nstates = 12

    def reset(self):
        R0 = np.array([[0.4830, 0.8365, 0.2588],
                       [-0.3209, 0.4441, -0.8365],
                       [-0.8147, 0.3209, 0.4830]])
        wb0 = np.array([[0], [0], [0]])
        X0 = np.vstack((R0.reshape(-1, 1), wb0.reshape(-1, 1)))
        s0 = X0.reshape(-1,)
        self.s0=s0
        return self.s0

    def reset_state(self, s):
        self.s0 = s
        return self.s0
    def hat(self,x):
        xhat = np.asarray([[0, -x[2].item(), x[1].item()], [x[2].item(), 0, -x[0].item()], [-x[1].item(), x[0].item(), 0]])
        return xhat
    def dynamics(self, y, t, u):
        r = y[0:9].reshape(-1, 3)
        w = y[9:12].reshape(-1, 1)
        dr = r @ self.hat(w)
        inv_I = np.linalg.inv(self.I)
        dw = np.dot(inv_I, -self.hat(w) @ self.I @ w + u.reshape(-1, 1))

        dr = dr.reshape(-1, 1)
        f = [dr[0].item(), dr[1].item(), dr[2].item(),
             dr[3].item(), dr[4].item(), dr[5].item(),
             dr[6].item(), dr[7].item(), dr[8].item(),
             dw[0].item(), dw[1].item(), dw[2].item()]
        return f

    def step(self, u):
        u = np.array(u).reshape(-1)
        sn = odeint(self.dynamics, self.s0, [0, self.dt], args=(u,))
        self.s0 = sn[-1, :]
        r = 0
        done = False
        return self.s0, r, done, {}

def hat( x):
    xhat = np.asarray([[0, -x[2].item(), x[1].item()], [x[2].item(), 0, -x[0].item()], [-x[1].item(), x[0].item(), 0]])
    return xhat

def getbasis(X, n):
    basis=np.empty((12 + 9 * n, 1))
    basis[0:12]=X.reshape(-1, 1)
    R=X[0:9].reshape(-1, 3)
    wb=X[9:12].reshape(-1, 1)
    wb_hat = hat(wb)
    Z = R
    for i in range(n):
        Z=Z@wb_hat
        basis[12+9*i:12+9*i+9]=Z.reshape(-1, 1)
    return basis

class data_collecter_so3():
    def __init__(self,env_name) -> None:
        self.env_name = env_name
        np.random.seed(2024)
        random.seed(2024)
        self.env = Spacecraft_so3()
        self.Nstates = self.env.Nstates
        self.udim = 3
        self.env.reset()
        self.dt = self.env.dt


    def collect_koopman_data(self,traj_num,steps,mode="train"):
        train_data = np.empty((steps+1,traj_num,self.Nstates+self.udim))
        for traj_i in range(traj_num):
            s0 = self.env.reset()
            u10 = np.random.uniform(-1,1, (3,))
            train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
            for i in range(1,steps+1):
                s0,r,done,_ = self.env.step(u10)
                u10 = np.random.uniform(-1,1, (3,))
                train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
        return train_data
