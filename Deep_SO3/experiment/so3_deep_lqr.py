import matplotlib.pyplot as plt
import os
import train as lka
from data import data_collecter_so3,getbasis
import torch
from scipy.spatial.transform import Rotation as R
from utils import matrixtoquaternion,getbasis,dynamics,dlqr,get_initialization_AB
from data import data_collecter_so3
import numpy as np
from scipy.integrate import odeint

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


#升维
def getlift(x):
    x=x.reshape(1, -1)
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
# R0= np.array([[0.74783335, -0.01635712, 0.66368477],
#                [0.40748675, 0.8005386, -0.43942187],
#                [-0.52411778, 0.59905713, 0.60533481]])
wb0 = np.array([[0],[0],[0]])

Rd = np.diag([1,1,1])
# Rd = np.array([[0.74783335, -0.01635712, 0.66368477],
#                [0.40748675, 0.8005386, -0.43942187],
#                [-0.52411778, 0.59905713, 0.60533481]])

# rd=np.array([[0.16],[0.8023],[-0.1579],[0.553]])
x0 = np.vstack((R0.reshape(-1, 1), wb0.reshape(-1, 1)))
x_ref_lift=np.vstack((Rd.reshape(-1,1),wb0.reshape(-1,1))).reshape(1,-1)
Ref=x_ref_lift
Xcurrent=x0

x0=getlift(x0)
x_ref_lift=getlift(x_ref_lift)


#提取A,B
Alift=net.lA.weight.data.cpu().numpy()
Blift=net.lB.weight.data.cpu().numpy()

nx=len(Alift)   #状态维度
nu=len(Blift.T) #状态维度
print(nx)

Q = np.zeros((nx, nx))
Q[:12, :12] = np.eye(12)

R=np.eye(nu)

k=dlqr(np.matrix(Alift),np.matrix(Blift),Q,R)
# k = lqr_regulator_k(np.matrix(Alift),np.matrix(Blift),Q,R)

#对比模拟情况'
# np.random.seed(2024)
# random.seed(2024)
# x_init_train = np.random.rand(7, 1) * 2 - 1
# x_init_train[0:4] = x_init_train[0:4] / np.linalg.norm(x_init_train[0:4])
# x_init_train[4:7] = x_init_train[4:7] * 0

steps=100
trajectories=np.empty((12,steps+1))
trajectories[:,0]=Xcurrent.reshape(-1,)
xkoop=np.empty((nx,steps+1))
x01=getlift(Xcurrent)
xkoop[:,0]=x01.reshape(-1,)
np.random.seed(3)
for i in range(steps):
    u = np.random.uniform(-1, 1, (3,))
    sn = odeint(dynamics, trajectories[:,i], [0, dt], args=(u,))
    trajectories[:, i+1] = sn[-1, :]

    # xx = getlift(trajectories[:,i]).reshape(-1,)
    # xkoop[:, i + 1] = Alift @ xx+ Blift @ u

    xkoop[:, i + 1] = Alift @ xkoop[:,i]+ Blift @ u
# for i in range(12):
#     plt.plot(trajectories[i,:steps],color='red')
#     plt.plot(xkoop[i, :steps])
#     plt.show()


# 可视化四元数q
qlift = np.empty((4, int(steps)))
q_true = np.empty((4, int(steps)))
for i in range(steps):
    qlift[:, i] = matrixtoquaternion(trajectories[:, i]).reshape(-1, )
    q_true[:,i]=matrixtoquaternion(xkoop[:,i]).reshape(-1,)
    # q_true[:, i] = matrixtoquaternion(Xd).reshape(-1, )

# 预测误差
fig, axs  =  plt.subplots(2, 2, figsize = (10, 12), sharex = True, gridspec_kw = {'height_ratios': [2, 2], 'width_ratios': [1, 1]})

for i in range(4):
    row, col  =  divmod(i, 2)
    axs[row, col].plot(np.linspace(0, steps, steps), qlift[i, :], label = f"Deep koopman ($q{i + 1}$)", color = 'red',marker = ".")
    axs[row, col].plot(np.linspace(0, steps, steps), q_true[i, :], label = f"True value ($q{i + 1}$)", color = 'blue',marker = "v",markersize=3)
    axs[row, col].set_xlabel('Step', family = 'Times New Roman', fontsize = 15)
    axs[row, col].set_ylabel(f"q$_{i + 1}$", family = 'Times New Roman', fontsize = 15, style = "italic")
    axs[row, col].xaxis.set_visible(True)


# 调整子图之间的间距
plt.subplots_adjust(hspace = 0.5)
# 在整体中下留白，避免与子图重叠
fig.legend(loc = 'lower center', bbox_to_anchor = (0.5, 0), prop = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15, }, ncol = 2, labels = ['Deep koopman', 'True value'])
# 调整布局
# plt.tight_layout(rect = [0, 1, 0, 1]) #左下右上 四个位置的偏移量 微调

# 垂直留白
plt.subplots_adjust(bottom = 0.15)
# 显示图形
plt.show()

fig, axs  =  plt.subplots(3, 1, figsize = (10, 12), sharex = True,)

for i in range(3):
    # row, col  =  divmod(i, 2)
    axs[i].plot(np.linspace(0, steps, steps), xkoop[i+9, :steps], label = f"Deep koopman ($q{i + 1}$)", color = 'red',marker = ".")
    axs[i].plot(np.linspace(0, steps, steps), trajectories[i+9, :steps], label = f"True value ($q{i + 1}$)", color = 'blue',marker = "v",markersize=3)
    axs[i].set_xlabel('Step', family = 'Times New Roman', fontsize = 15)
    axs[i].set_ylabel(chr(969)+f"$_{i + 1}$", family = 'Times New Roman', fontsize = 15, style = "italic")
    # axs[row, col].xaxis.set_visible(True)


# 调整子图之间的间距
plt.subplots_adjust(hspace = 0.5)
# 在整体中下留白，避免与子图重叠
fig.legend(loc = 'lower center', bbox_to_anchor = (0.5, 0), prop = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15, }, ncol = 2, labels = ['Deep koopman', 'True value'])
# 调整布局
# plt.tight_layout(rect = [0, 1, 0, 1]) #左下右上 四个位置的偏移量 微调

# 垂直留白
plt.subplots_adjust(bottom = 0.15)
# 显示图形
plt.show()


# for i in range(4):
#     plt.plot(np.linspace(0, steps, steps), qlift[i, :], label="Ture", color='red')
#     plt.plot(np.linspace(0, steps, steps), q_true[i, :], label="Ture")
#     plt.legend(loc='lower left')
#     plt.title("True_and_koopman", fontsize=15)
#     plt.xlabel("$t$", fontsize=13, x=1)
#     plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
#     plt.show()
#
# #
steps=800
Turedata = np.zeros((12, steps+1))
Turedata[:,0]=Xcurrent.reshape(-1,)
u_data = np.zeros((3, steps+1))
for i in range(steps):
    # x_ref_lift=getbasis(trajectories[:, i+1],n_basis)
    u = -k@(x0-x_ref_lift)
    u_data[:,i]=u.reshape(-1,)
    # u= np.clip(u, -1, 1)
    sn = odeint(dynamics, Xcurrent.reshape(-1,), [0, dt], args=(u,))
    Xcurrent=sn[-1, :]
    Turedata[:, i+1] = Xcurrent
    x0=getlift(Xcurrent)

# for i in range(12):
#     plt.plot(Turedata[i, :], label="x2",color='red')
#     # plt.plot(trajectories[i,:steps])
#     # plt.axhline(Ref[i])
#     plt.show()

errors = np.zeros((3, steps))
for i in range(steps):
    R=Turedata[:9, i].reshape(3,3)
    e=(Rd@R-R.T@Rd)
    errors[0,i]=e[1,2]
    errors[1, i] = e[0,2]
    errors[2, i] = e[1,0]

### 跟踪误差
# plt.xlabel('Time (s)',family='Times New Roman',fontsize = 15)
# plt.ylabel('e',family='Times New Roman',fontsize = 20,fontweight= 'bold',style= "italic")
# y=np.linspace(0,steps/10,steps)
# plt.plot(y,errors[0, :], label="e$_1$", color='blue')
# plt.plot(y,errors[1, :], label="e$_2$", color='y')
# plt.plot(y,errors[2, :], label="e$_3$", color='red')
# font_legend = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,          'style': "italic",
#          }
# plt.legend(loc='upper right',prop = font_legend)
# plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.xticks(fontproperties = 'Times New Roman', size = 14)
# # plt.savefig("test1.svg", dpi=300,format="svg")
# plt.savefig("test1")
# plt.show()

### 角速度
# plt.xlabel('Time (s)',family='Times New Roman',fontsize = 15)
# plt.ylabel(chr(969),family='Times New Roman',fontsize = 20,fontweight= 'bold',style= "italic")
# y=np.linspace(0,steps/10,steps)
# plt.plot(y,Turedata[9, :steps], label=chr(969)+"$_1$", color='blue')
# plt.plot(y,Turedata[10, :steps], label=chr(969)+"$_2$", color='y')
# plt.plot(y,Turedata[11, :steps], label=chr(969)+"$_3$", color='red')
# font_legend = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,          'style': "italic",
#          }
# plt.legend(loc='upper right',prop = font_legend)
# plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.xticks(fontproperties = 'Times New Roman', size = 14)
# # plt.savefig("test1.svg", dpi=300,format="svg")
# plt.savefig("w_sudu")
# plt.show()

### 控制量
# plt.xlabel('Time (s)',family='Times New Roman',fontsize = 15)
# plt.ylabel(chr(964),family='Times New Roman',fontsize = 20,fontweight= 'bold',style= "italic")
# y=np.linspace(0,steps/10,steps)
# plt.plot(y,u_data[0, :steps], label=chr(964)+"$_1$", color='blue')
# plt.plot(y,u_data[1, :steps], label=chr(964)+"$_2$", color='y')
# plt.plot(y,u_data[2, :steps], label=chr(964)+"$_3$", color='red')
# font_legend = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,          'style': "italic",
#          }
# plt.legend(loc='upper right',prop = font_legend)
# plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.xticks(fontproperties = 'Times New Roman', size = 14)
# # plt.savefig("test1.svg", dpi=300,format="svg")
# plt.savefig("t_")
# plt.show()
