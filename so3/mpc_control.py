import numpy as np
import casadi
from scipy.integrate import odeint
from utils import matrixtoquaternion,hat,getbasis,dynamics,dm_to_array
import matplotlib.pyplot as plt
from scipy import linalg as lin
from utils import dlqr
def lqr_p_k_cal(a: np.matrix, b: np.matrix, q: np.matrix, r: np.matrix):
    p = np.mat(lin.solve_discrete_are(a, b, q, r))  # 求Riccati方程
    k = lin.inv(r + b.T * p * b) * b.T * p * a

    return p, k


# 初始状态R0与参考状态Rd

# R0=np.array([[0.74783335, -0.01635712,  0.66368477],
#              [0.40748675,  0.8005386,  -0.43942187],
#             [-0.52411778,  0.59905713,  0.60533481]])
R0=np.array([[0.4830, 0.8365,  0.2588],
             [-0.3209,   0.4441,  -0.8365],
            [-0.8147,  0.3209,  0.4830]])

wb0 = np.array([[0],[0],[0]])
X0=np.vstack((R0.reshape(-1,1),wb0.reshape(-1,1)))
Rd = np.diag([1,1,1])
Xd=np.vstack((Rd.reshape(-1,1),wb0.reshape(-1,1)))

Alift=np.load('Alift.npy')
Blift=np.load('Blift.npy')

##  初始化变量
nx=len(Alift)   #状态维度
nu=len(Blift.T) #状态维度
N=50        #预测时域4

Q = np.zeros((nx, nx))
Q[:12, :12] = np.eye(12)*15
# Q = np.eye(nx)*15
R=np.eye(nu)
n_basis=3
dt=0.1

#######MPC

states = casadi.SX.sym('states',nx)
n_states = states.numel()

controls = casadi.SX.sym('controls',nu)
n_controls = controls.numel()

X = casadi.SX.sym('X', n_states, N + 1)
U = casadi.SX.sym('U', n_controls, N)
P = casadi.SX.sym('P', 2 * n_states)

A = casadi.SX(np.matrix(Alift))  # 状态矩阵
B = casadi.SX(np.matrix(Blift))  # 状态矩阵

#变换方程
st_fun_nom = A @ states + B @ controls
f_nom = casadi.Function('f_nom', [states, controls], [st_fun_nom])  # 对应状态方程中的f()

cost = 0
g = X[:, 0] - P[:n_states]

p, k1 =lqr_p_k_cal(np.matrix(Alift), np.matrix(Blift),np.matrix(Q),np.matrix(R))
k2=dlqr(np.matrix(Alift),np.matrix(Blift),Q,R)


for k in range(N):
    state = X[:, k]
    control = U[:, k]
    cost = cost + (state - P[n_states:]).T @ Q@ (state - P[n_states:])+ control.T @ R @ control
    next_state = X[:, k + 1]
    predicted_state = f_nom(state, control)
    g = casadi.vertcat(g, next_state - predicted_state)

# cost=cost++ 1 / 2 * (X[:, N-1]-P[n_states:]).T @ p @ (X[:, N-1]-P[n_states:]) # 还应加上终端代价函数
opt_variables = casadi.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

nlp_prob = {
    'f': cost,
    'x': opt_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'sb': 'yes',
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0,
}
solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = casadi.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
ubx = casadi.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

for i in range(n_states):
    lbx[i:n_states * (N + 1):n_states] = -casadi.inf
    ubx[i:n_states * (N + 1):n_states] = casadi.inf

# for i in range(3):
#     lbx[9+i:n_states * (N + 1):n_states] = -0.01
#     ubx[9+i:n_states * (N + 1):n_states] = 0.01


for i in range(n_controls):
    lbx[n_states * (N + 1)+i:n_states * (N + 1) + n_controls * N:n_controls] = -1
    ubx[n_states * (N + 1)+i:n_states * (N + 1) + n_controls * N:n_controls] =1

lbg = casadi.DM.zeros((n_states * (N + 1) ))
ubg = casadi.DM.zeros((n_states * (N + 1)))

args = {
    'lbg': lbg,
    'ubg': ubg,
    'lbx': lbx,
    'ubx': ubx
}

Xcurrent=X0.reshape(-1,)
x0=getbasis(X0,n_basis)
x_ref_lift=getbasis(Xd,n_basis)
state_0 = casadi.DM(x0)
state_ref = casadi.DM(x_ref_lift)

u0 = casadi.DM.zeros((n_controls, N))
X0 = casadi.repmat(state_0, 1, N + 1)

State=dm_to_array(state_0)

n_normal=12 #12 代表状态维度（未提升）

# 得到下一时刻的状态
def shift_timestep( state, control, f):
    next_state = f(state, control[:, 0])
    next_control = casadi.horzcat(control[:, 1:],
                                  casadi.reshape(control[:, -1], -1, 1))
    return  next_state, next_control
if __name__ == '__main__':
    Ns = 1200
    Turedata = np.zeros((n_normal, Ns + 1))
    Turedata[:,0]=dm_to_array(state_0[:n_normal]).reshape(-1,)
    controldata = np.zeros((nu, Ns))

    for i in range(Ns):
        print(i)
        args['p'] = casadi.vertcat(state_0, state_ref)
        args['x0'] = casadi.vertcat(casadi.reshape(X0, n_states * (N + 1), 1),
                                    casadi.reshape(u0, n_controls * N, 1))
        sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                     lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = casadi.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = casadi.reshape(sol['x'][:n_states * (N + 1)], n_states, N + 1)

        state_0, u0 = shift_timestep(state_0, u, f_nom) #state_0[39,1] u0[3 10]
        State=np.hstack((State,state_0))
        # 第一个控制量
        uu = np.array(u0[:, 0]).reshape(-1, 1)
        controldata[:, i] = uu.reshape(-1, )
        #真实状态转移
        sn = odeint(dynamics, Xcurrent, [0, dt], args=(uu,))
        Xcurrent=sn[-1, :]
        state_0 = np.matrix(getbasis(Xcurrent,n_basis)).reshape(-1, 1)

        Turedata[:,i+1]=Xcurrent.reshape(-1,)
        X0 = casadi.horzcat(X0[:, 1:], casadi.reshape(X0[:, -1], -1, 1))

# for i in range(12):
#     plt.plot(State[i,:], label="x1")
#     plt.plot(Turedata[i, :], label="x2")
#     plt.axhline(Xd[i])
#     plt.show()

# fig,ax=plt.subplots(3,1)

# 可视化四元数q
steps=Ns
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