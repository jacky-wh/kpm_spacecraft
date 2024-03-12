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
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def load_A_B():
    Alift = np.load('C:/Users/admin/Desktop/Deep_koopman (2)/Deep_koopman3/Alift.npy')
    Blift = np.load('C:/Users/admin/Desktop/Deep_koopman (2)/Deep_koopman3/Blift.npy')
    Alift = torch.from_numpy(Alift)
    Blift = torch.from_numpy(Blift)
    return Alift,Blift

class Network(nn.Module):
    def __init__(self, encode_layers, Nkoopman, u_dim):

        super(Network, self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
            if layer_i != len(encode_layers) - 2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)

        Alift,Blift=load_A_B()
        Nkoopman=Alift.shape[0]
        self.Nkoopman = Alift.shape[0]
        self.u_dim = Blift.shape[1]
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = Alift
        self.lB = nn.Linear(self.u_dim, Nkoopman, bias=False)
        self.lB.weight.data = Blift

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
    X_current = net.encode(data[0, :, u_dim:u_dim+12])
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1, dtype=torch.float64).to(device)
    loss1 = torch.zeros(1, dtype=torch.float64).to(device)
    loss2 = torch.zeros(1, dtype=torch.float64).to(device)
    Augloss = torch.zeros(1, dtype=torch.float64).to(device)
    for i in range(steps - 1):
        X_current = net.forward(X_current, data[i, :, :u_dim])
        beta_sum += beta
        Y = data[i + 1, :, u_dim:]
        loss += beta * mse_loss(X_current, Y)
        Y=net.encode(data[i + 1, :, u_dim:u_dim + 12])
        loss += beta * mse_loss(X_current, Y)
        X_current = Y
    #     if not all_loss:
    #         # loss += beta * mse_loss(X_current[:, :Nstate], data[i + 1, :, u_dim:])
    #         loss1+= beta * mse_loss(X_current[:, :9], data[i + 1, :, u_dim:u_dim+9])
    #         loss2 += beta * mse_loss(X_current[:, 9:12], data[i + 1, :, u_dim+9:u_dim+12])
    #     else:
    #         Y = net.encode(data[i + 1, :, u_dim:])
    #         loss += beta * mse_loss(X_current, Y)
    #     X_current_encoded = net.encode(X_current[:, :Nstate])
    #     Augloss += mse_loss(X_current_encoded, X_current)
    #     beta *= gamma
    # loss = 200 * loss1 + 400 * loss2
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
    in_dim = data_collect.Nstates
    Nstate = in_dim
    # layer_depth = 4
    layer_width = 128
    encode_dim=Ktest_data.shape[-1] - u_dim-in_dim
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
    logdir = "./Data/" + suffix + "/KoopmanU_" + env_name
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

        loss = Kloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train/loss', loss, i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i + 1) % eval_step == 0:
            # K loss
            with torch.no_grad():
                Kloss = Klinear_loss(Ktest_data, net, mse_loss, u_dim, gamma, Nstate, all_loss=0)
                loss = Kloss
                Kloss = Kloss.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy()

                # A,B=net.AB()
                # mm=kekong(A.cpu().numpy(),B.cpu().numpy())
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
    parser.add_argument("--suffix",type=str,default="2-29")
    parser.add_argument("--all_loss",type=int,default=0)
    parser.add_argument("--K_train_samples",type=int,default=50000)
    parser.add_argument("--e_loss",type=int,default=0)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--encode_dim",type=int,default=20)
    parser.add_argument("--layer_depth",type=int,default=3)
    args = parser.parse_args()
    main()

