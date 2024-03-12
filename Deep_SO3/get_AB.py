import numpy as np
from utils import get_initialization_AB
from data import data_collecter_so3
import os
import torch
# import trainnet as lka
import train as lka
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
#getab
Alift,Blift=get_initialization_AB()
np.save('Alift',Alift)
np.save('Blift',Blift)

#getdnnab

#加载模型
suffix = "2-29"
env_names = "Spacecraft_attitude"
method = 'KoopmanU'
root_path = "./Data/" + suffix
for file in os.listdir(root_path):
    if file.startswith(method + "_" + env_names) and file.endswith(".pth"):
        model_path = file

Data_collect = data_collecter_so3(env_names)
udim = Data_collect.udim
Nstates = Data_collect.Nstates

dicts = torch.load(root_path+"/"+model_path,map_location=torch.device('cpu'))

state_dict = dicts["model"]

net = lka.Network()
net.load_state_dict(state_dict)
# device = torch.device("cuda:0")
net.cuda()
net.double()

#提取A,B
Alift=net.lA.weight.data.cpu().numpy()
Blift=net.lB.weight.data.cpu().numpy()

np.save('Alift_dnn',Alift)
np.save('Blift_dnn',Blift)

#getdata
# env_name='test'
# Ktest_samples=10
# Ktrain_samples=10
# Ksteps=300
#
# data_collect = data_collecter_so3(1)
# test_data_path = "./Data/dataset/test/{}.npy".format(env_name)
# train_data_path = "./Data/dataset/train/{}.npy".format(env_name)
# if os.path.exists(test_data_path):
#     Ktest_data = np.load("./Data/dataset/test/{}.npy".format(env_name))
# else:
#     Ktest_data = data_collect.collect_koopman_data(Ktest_samples, Ksteps, mode="eval")
#     np.save("./Data/dataset/test/{}.npy".format(env_name), Ktest_data)
# print("test data ok!,shape:", Ktest_data.shape)
# if os.path.exists(train_data_path):
#     Ktrain_data = np.load("./Data/dataset/train/{}.npy".format(env_name))
# else:
#     Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples, Ksteps, mode="train")
#     np.save("./Data/dataset/train/{}.npy".format(env_name), Ktrain_data)
# print("train data ok!,shape:", Ktrain_data.shape)
