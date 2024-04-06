import numpy as np
from scipy.integrate import odeint
from utils import matrixtoquaternion,getbasis,dynamics,lqr_regulator_k,quaternion2rot,dlqr,lqr_p_k_cal
from scipy.linalg import sqrtm
from cvxpy import Variable
import cvxpy as cp
import matplotlib.pyplot as plt


def DRMPCmpcgain(AK, Bd, G, K, q, r, P, N):
    m, n = Bd.shape
    m1, n1 = q.shape
    m2, n2 = r.shape
    m3, n3 = AK.shape
    m4, n4 = G.shape
    m5, n5 = K.shape


    # 公式（9）中的黑体A
    A_bf_cell = np.zeros(((N+1)*m3, n3))
    A_bf_cell[:m3] = np.eye(m3, n3)
    for kk in range(0, N):
        A_bf_cell[(kk+1)*m3 : (kk+2)*m3] = A_bf_cell[kk*m3 : (kk+1)*m3] @ AK

    # 公式（9）中的黑体B
    # 公式（9）中的黑体B
    B_bf_cell = np.zeros(((N+1)*m, N*n))
    for i in range(N):
        B_bf_cell[(i+1)*m : (i+2)*m, i*n : (i+1)*n] = Bd
    for i in range(1,N):
        for j in range(i+1, N+1):
            B_bf_cell[j*m : (j+1)*m, (i-1)*n : (i)*n] = AK @ B_bf_cell[(j-1)*m : j*m,(i-1)*n : (i)*n]

    # 公式（9）中的黑体G
    G_bf_cell = np.zeros(((N+1)*m4, N*n4))
    for i in range(N):
        G_bf_cell[(i+1)*m4 : (i+2)*m4, i*n4 : (i+1)*n4] = G
    for i in range(1,N):
        for j in range(i+1, N+1):
            G_bf_cell[(j)*m4 : (j+1)*m4, (i-1)*n4 : (i)*n4] = AK @ G_bf_cell[(j-1)*m4 : (j)*m4, (i-1)*n4 : (i)*n4]

    # 公式（31）中的黑体Q,R
    Q_cell = np.zeros(((N+1)*m1, (N+1)*n1))
    for k1 in range(N+1):
        for k2 in range(N+1):
            if k1 == k2:
                Q_cell[k1*m1 : (k1+1)*m1, k2*n1 : (k2+1)*n1] = q
    R_cell = np.zeros((N*m2, N*n2))
    for k1 in range(N):
        for k2 in range(N):
            if k1 == k2:
                R_cell[k1*m2 : (k1+1)*m2, k2*n2 : (k2+1)*n2] = r

    K_bf_cell = np.zeros((N*m5, (N+1)*n5))
    for k1 in range(N):
        for k2 in range(N+1):
            if k1 == k2:
                K_bf_cell[k1*m5 : (k1+1)*m5, k2*n5 : (k2+1)*n5] = K

    Q_cell[N*m1 : (N+1)*m1, N*n1 : (N+1)*n1] = P

    return A_bf_cell, B_bf_cell, G_bf_cell, K_bf_cell, Q_cell, R_cell


#由非0终端点xref引出的量
def DRMPCmpcgain2(xref,B,k,N,Q_N,B_bf,K_bf,R_N,AK):
    m3, n3 = AK.shape

    # 公式（9）中的黑体B
    B_bf_cell = np.zeros(((N+1)*m3, N*m3))
    for i in range(N):
        B_bf_cell[(i+1)*m3 : (i+2)*m3, i*m3 : (i+1)*m3] = np.eye(m3)
    for i in range(1,N):
        for j in range(i+1, N+1):
            B_bf_cell[j*m3 : (j+1)*m3, (i-1)*m3 :(i)*m3] = AK @ B_bf_cell[(j-1)*m3 : j*m3,(i-1)*m3 : (i)*m3]

    # 公式（9）中的黑体A
    F_cell = np.zeros(((N+1)*m3, n3))
    F_cell[m3:2*m3] = np.eye(m3, n3)
    for kk in range(1, N):
        F_cell[(kk+1)*m3 : (kk+2)*m3] = F_cell[kk*m3 : (kk+1)*m3] @ AK

    xref=xref.reshape(-1,1)
    xref_bk=B@k@xref
    # xref_bk_bf=np.tile(xref_bk, (N + 1, 1))
    # xref_bk_bf=F_cell@xref_bk
    xref_bk_bf=np.tile(xref_bk, (N , 1))
    xref_bk_bf=B_bf_cell@xref_bk_bf
    o1 = -2 * xref_bk_bf.T @ Q_N @ B_bf

    xref_f = np.tile(xref, (N + 1, 1))
    o2 = -2 * xref_f.T @ Q_N @ B_bf

    S1 = K_bf @ xref_bk_bf
    S2 = K_bf @ xref_f
    S = S1 + S2
    o3 = -2 * S.T @ R_N @ (K_bf @ B_bf+np.eye(len(K_bf)))
    return o1+o2+o3

def DRMPCcostfunction(A_bf, B_bf, G_bf, K_bf, Q, R, N, xm,xref,B,k):
    ch_delta1 = 0.1
    ch_delta2 = 0.15
    lam1 = 0
    lam2 = 1
    ch_delta = lam1 * (ch_delta1)**2 + lam2 * (ch_delta2)**2
    Omega1 = ch_delta * np.eye(len(xm)*N)

    m, n = K_bf.shape
    xm=xm.reshape(-1,1)
    g_bf=2*(A_bf@xm).T@Q@B_bf+2*(A_bf@xm).T@ K_bf.T@R@(K_bf@B_bf+np.eye(m))
    # g_bf = (2 * np.dot(np.dot(A_bf.T, Q), B_bf) + 2 * np.dot(np.dot(xm.T, np.dot(A_bf.T, K_bf.T)), np.dot(R, K_bf.dot(B_bf) + np.eye(m))))
    # print(g_bf.shape)
    # c = (A_bf @ xm).T @ Q @ (A_bf @ xm) + xm.T @ A_bf.T @ K_bf.T @ R @ (K_bf @ A_bf @ xm) + np.trace(
    #     G_bf.T @ Q @ G_bf @ Omega1) + np.trace(G_bf.T @ K_bf.T @ R @ K_bf @ G_bf @ Omega1)


    delta_bf = (np.dot(B_bf.T, Q.dot(B_bf)) + np.dot((np.dot(K_bf, B_bf) + np.eye(m)).T, np.dot(R, K_bf.dot(B_bf) + np.eye(m))))

    epsilon = 0.3
    nx=len(xm)
    a_bf = np.zeros((nx*N+nx, N))
    for j in range(N):
        a_bf[7*(j+1)+4, j] = 1

    c_bf = np.eye(N)
    c1_bf = -np.eye(N)

    d = 2
    b = 2

    v = Variable(N*nu)
    t = Variable()

    constraints = []
    for j in range(N-1):
        a_bf_j=a_bf[:, j].reshape(-1,1)
        constraints += [
            cp.sqrt((1 - epsilon) / epsilon) * cp.norm(
                cp.multiply(a_bf_j.T @ G_bf, sqrtm(Omega1))) <= b - np.matmul(
                a_bf_j.T, (A_bf @ xm.reshape(-1,) + B_bf @ v)),
            # cp.norm(cp.sqrt((1 - epsilon) / epsilon) * (c_bf[:, j].T @ K_bf @ G_bf @ sqrtm(Omega1))) <= d - c_bf[:,
            #                                                                                                    j].T @ (
            #             K_bf @ A_bf @ xm + K_bf @ B_bf @ v + v),
            # cp.norm(cp.sqrt((1 - epsilon) / epsilon) * (c1_bf[:, j].T @ K_bf @ G_bf @ sqrtm(Omega1))) <= d - c1_bf[:,
            #                                                                                                     j].T @ (
            #             K_bf @ A_bf @ xm + K_bf @ B_bf @ v + v)
        ]
    o3=DRMPCmpcgain2(xref,B,k,N,Q,B_bf,K_bf,R,AK)

    constraints += [cp.abs(v) <= 1]


    constraints += [
        cp.quad_form(v, delta_bf) + g_bf @ v + o3@ v<= t
    ]

    objective = cp.Minimize(t)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    u_m = v.value[:3]
    flag = problem.status

    return flag, u_m

R0=np.array([[0.4830, 0.8365,  0.2588],
             [-0.3209,   0.4441,  -0.8365],
            [-0.8147,  0.3209,  0.4830]])
# R0=np.array([[0.74783335, -0.01635712,  0.66368477],
#              [0.40748675,  0.8005386,  -0.43942187],
#             [-0.52411778,  0.59905713,  0.60533481]])

wb0 = np.array([[0],[0],[0]])
X0=np.vstack((R0.reshape(-1,1),wb0.reshape(-1,1)))
dt=0.1
Rd = np.diag([1,1,1])
Xd=np.vstack((Rd.reshape(-1,1),wb0.reshape(-1,1)))
n_basis=3
Xcurrent=X0

A=np.load('Alift.npy')
B=np.load('Blift.npy')

##  初始化变量
nx=len(A)   #状态维度
nu=len(B.T) #状态维度
# G = np.eye(nx)
G= np.zeros((nx, nx))
G[:7,:7]=np.eye(7)
Q = np.zeros((nx, nx))
Q[:12, :12] = np.eye(12)*10
R=np.eye(nu)
k=-dlqr(np.matrix(A),np.matrix(B),Q,R)



AK = A +B @ k
N=50
A_bf, B_bf, G_bf, K_bf, Q, R = DRMPCmpcgain(AK, B, G, k, Q, R, Q, N)


M = 800
M1 = 1

DRMPCx_obj = np.zeros((nx, M+1, M1))  # 扰动系统状态
DRMPCx_star = np.zeros((nx, M+1, M1))  # 标称系统状态
DRMPCv_star = np.zeros((nu, M+1, M1))  # 标称系统控制
DRMPCu_obj = np.zeros((nu, M+1, M1))  # 标称系统控制

x0=getbasis(X0,n_basis)
xref=getbasis(Xd,n_basis)

Turedata = np.zeros((12, M+1))
Turedata[:,0]=Xcurrent.reshape(-1,)
for j in range(M1):
    DRMPCx_star[:, 0, j] = x0.reshape(-1,)
    DRMPCx_obj[:, 0, j] = x0.reshape(-1,)
    flag1 = 'Solved'
    flag2 = 'Infeasible'
    i = 0
    while i < M :
        print(j, i)
        flag1, DRMPCv_star[:, i, j] = DRMPCcostfunction(A_bf, B_bf, G_bf, K_bf, Q, R, N, DRMPCx_obj[:, i, j],xref,B,k)
        if i>300:
            DRMPCv_star[:, i, j] = 0

        # DRMPCv_star[:, i, j]=0
        flag = flag1 == flag2
        if flag:
            DRMPCx_star[:, i, j] = AK @ DRMPCx_star[:, i-1, j] + Bd @ DRMPCv_star[:, i-1, j]
            i -= 1
        else:
            # ch_delta1 = 0.1
            # ch_delta2 = 0.15
            # lam1 = 0.4
            # lam2 = 0.6
            # w = lam1 * np.random.normal(0, ch_delta1, [2, 1]) + lam2 * np.random.normal(0, ch_delta2, [2, 1])
            # w=np.clip(w, -0.1, 0.1)
            # w=w*0

            DRMPCu_obj[:, i+1, j] = k @ ((DRMPCx_obj[:, i, j].reshape(-1, 1))-xref).reshape(-1,) + DRMPCv_star[:, i, j].reshape(-1,)
            sn = odeint(dynamics, Xcurrent.reshape(-1, ), [0, dt], args=( DRMPCu_obj[:, i+1, j].reshape(-1,),))
            Xcurrent = sn[-1, :]
            Turedata[:, i + 1] = Xcurrent
            DRMPCx_obj[:, i + 1, j] = getbasis(Xcurrent, n_basis).reshape(-1, )
            print('v',DRMPCv_star[:, i, j])
            print('DRMPCu_obj', DRMPCu_obj[:, i+1, j])
        i += 1

steps=M
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