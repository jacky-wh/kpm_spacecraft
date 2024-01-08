import numpy as np
from scipy.spatial.transform import Rotation
import scipy.linalg as la

def dlqr_with_arimoto_potter(Ad, Bd, Q, R):
    dt=0.1
    """Solve the discrete time lqr controller.
    x[k+1] = Ad x[k] + Bd u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    n = len(Bd)

    # continuous
    Ac = (Ad - np.eye(n)) / dt
    Bc = Bd / dt

    # Hamiltonian
    Ham = np.vstack(
        (np.hstack((Ac, - Bc * la.inv(R) * Bc.T)),
         np.hstack((-Q, -Ac.T))))

    eigVals, eigVecs = la.eig(Ham)

    V1 = None
    V2 = None

    for i in range(2 * n):
        if eigVals[i].real < 0:
            if V1 is None:
                V1 = eigVecs[0:n, i]
                V2 = eigVecs[n:2 * n, i]
            else:
                V1 = np.vstack((V1, eigVecs[0:n, i]))
                V2 = np.vstack((V2, eigVecs[n:2 * n, i]))
    V1 = np.matrix(V1.T)
    V2 = np.matrix(V2.T)

    P = (V2 * la.inv(V1)).real

    K = la.inv(R) * Bc.T * P

    return K
def lqr_regulator_k(A,B,Q,R):
    Kopt = dlqr_with_arimoto_potter(A, B, Q, R)
    return Kopt


def matrixtoquaternion(R):
    R=R[0:9].reshape(3,3)
    r = Rotation.from_matrix(R)
    quaternion = r.as_quat()
    return quaternion

def hat( x):
    xhat = np.asarray([[0, -x[2].item(), x[1].item()], [x[2].item(), 0, -x[0].item()], [-x[1].item(), x[0].item(), 0]])
    return xhat

def dynamics( y, t, u):
    I = np.diag([0.1, 0.2, 0.3])
    r = y[0:9].reshape(-1, 3)
    w = y[9:12].reshape(-1, 1)
    dr=r @ hat(w)
    inv_I = np.linalg.inv(I)
    dw= np.dot(inv_I, -hat(w) @ I @ w + u.reshape(-1, 1))

    dr=dr.reshape(-1,1)
    f = [dr[0].item(), dr[1].item(), dr[2].item(),
         dr[3].item(), dr[4].item(), dr[5].item(),
         dr[6].item(), dr[7].item(), dr[8].item(),
         dw[0].item(), dw[1].item(), dw[2].item()]
    return f

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

def dm_to_array(dm):
    return np.array(dm.full())

