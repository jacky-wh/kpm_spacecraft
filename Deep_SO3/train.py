import torch
import numpy as np
import torch.nn as nn
import random
from collections import OrderedDict
from copy import copy
import argparse
from torch.utils.tensorboard import SummaryWriter
from data import data_collecter_so3
import time
from scipy.integrate import odeint
import os
from utils import matrixtoquaternion,getbasis,dynamics,lqr_regulator_k,quaternion2rot,dlqr,lqr_p_k_cal
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega

class Network(nn.Module):
    def __init__(self, encode_layers, Nkoopman, u_dim):
        super(Network, self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
            if layer_i != len(encode_layers) - 2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim, Nkoopman, bias=False)


    def encode_only(self, x):
        return self.encode_net(x)
    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], axis=-1)
    def forward(self, x, u):
        return self.lA(x) + self.lB(u)


def Klinear_loss(data, net, mse_loss, u_dim=1, gamma=0.99, Nstate=4, all_loss=0):
    steps, train_traj_num, NKoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_current = net.encode(data[0, :, u_dim:])
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1, dtype=torch.float64).to(device)
    loss1 = torch.zeros(1, dtype=torch.float64).to(device)
    loss2 = torch.zeros(1, dtype=torch.float64).to(device)
    Augloss = torch.zeros(1, dtype=torch.float64).to(device)
    for i in range(steps - 1):
        X_current = net.forward(X_current, data[i, :, :u_dim])
        beta_sum += beta
        Y = net.encode(data[i + 1, :, u_dim:])
        loss1 += beta * mse_loss(X_current, Y)
        loss2 += beta * mse_loss(X_current[:,:12], Y[:,:12])
        loss=loss1+10*loss2
    loss = loss / beta_sum
    Augloss = Augloss / beta_sum
    return loss + 0.5 * Augloss

def train(env_name, train_steps=200000, suffix="", all_loss=0, \
          encode_dim=12, layer_depth=3, e_loss=1, gamma=1, Ktrain_samples=50000):
    Ktrain_samples = Ktrain_samples
    Ktest_samples = 20000
    Ksteps = 30
    Kbatch_size = 100
    res = 1
    normal = 1
    # data prepare

    data_collect = data_collecter_so3(env_name)
    u_dim = data_collect.udim
    test_data_path="./Data/dataset/test/{}.npy".format(env_name)
    train_data_path = "./Data/dataset/train/{}.npy".format(env_name)
    if os.path.exists(test_data_path):
        Ktest_data = np.load("./Data/dataset/test/{}.npy".format(env_name))
    else:
        Ktest_data = data_collect.collect_koopman_data(Ktest_samples, Ksteps, mode="eval")
        np.save("./Data/dataset/test/{}.npy".format(env_name), Ktest_data)
    print("test data ok!,shape:", Ktest_data.shape)
    if os.path.exists(train_data_path):
        Ktrain_data = np.load("./Data/dataset/train/{}.npy".format(env_name))
    else:
        Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples, Ksteps, mode="train")
        np.save("./Data/dataset/train/{}.npy".format(env_name), Ktrain_data)
    print("train data ok!,shape:", Ktrain_data.shape)

    Ktrain_samples = Ktrain_data.shape[1]
    in_dim = Ktest_data.shape[-1] - u_dim
    Nstate = in_dim
    # layer_depth = 4
    layer_width = 128
    layers = [in_dim] + [layer_width] * layer_depth + [encode_dim]
    Nkoopman = in_dim + encode_dim
    print("layers:", layers)
    net = Network(layers, Nkoopman, u_dim)
    # print(net.named_modules())
    eval_step = 1000
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda()
    net.double()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate)
    for name, param in net.named_parameters():
        print("model:", name, param.requires_grad)
    # train
    eval_step = 1000
    best_loss = 1000.0
    best_state_dict = {}
    logdir = "./Data/" + suffix + "/KoopmanU_" + env_name + "layer{}_edim{}_eloss{}_gamma{}_aloss{}_samples{}".format(
        layer_depth, encode_dim, e_loss, gamma, all_loss, Ktrain_samples)
    if not os.path.exists("./Data/" + suffix):
        os.makedirs("./Data/" + suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    start_time = time.process_time()
    for i in range(train_steps):
        # K loss
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:, Kindex[:Kbatch_size], :]
        Kloss = Klinear_loss(X, net, mse_loss, u_dim, gamma, Nstate, all_loss)
        # loss = Kloss + Eloss+ KKloss1
        loss = Kloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train/Kloss', Kloss, i)
        writer.add_scalar('Train/loss', loss, i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i + 1) % eval_step == 0:
            # K loss
            with torch.no_grad():
                Kloss = Klinear_loss(Ktest_data, net, mse_loss, u_dim, gamma, Nstate, all_loss=0)
                loss = Kloss
                loss = loss.detach().cpu().numpy()
                Kloss = Kloss.detach().cpu().numpy()
                writer.add_scalar('Eval/loss', loss, i)

                if loss < best_loss :
                    best_loss = copy(Kloss)
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {'model': best_state_dict, 'layer': layers}
                    torch.save(Saved_dict, logdir + ".pth")
                print("Step:{} Eval-loss{} K-loss:{} ".format(i, loss, Kloss))
            # print("-------------END-------------")
        writer.add_scalar('Eval/best_loss', best_loss, i)
        # if (time.process_time()-start_time)>=210*3600:
        #     print("time out!:{}".format(time.clock()-start_time))
        #     break
    print("END-best_loss{}".format(best_loss))
def main():
    train(args.env, suffix=args.suffix, all_loss=args.all_loss, \
          encode_dim=args.encode_dim, layer_depth=args.layer_depth, \
          e_loss=args.e_loss, gamma=args.gamma, \
          Ktrain_samples=args.K_train_samples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="Spacecraft_attitude")
    parser.add_argument("--suffix",type=str,default="3-7")
    parser.add_argument("--all_loss",type=int,default=0)
    parser.add_argument("--K_train_samples",type=int,default=10000)
    parser.add_argument("--e_loss",type=int,default=0)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--encode_dim",type=int,default=40)
    parser.add_argument("--layer_depth",type=int,default=3)
    args = parser.parse_args()
    main()

