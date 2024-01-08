# 导入相关包

import math

import sys

import os

import matplotlib.pyplot as plt

import numpy as np

import scipy.linalg as la

# cubic_spline_planner为自己实现的三次样条插值方法


# 实现离散Riccati equation 的求解方法

def solve_dare(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """

    x = Q

    x_next = Q

    max_iter = 150

    eps = 0.01

    for i in range(max_iter):
        x_next = A.T @ x @ A - A.T @ x @ B @ \
 \
                 la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q

        if (abs(x_next - x)).max() < eps:
            break

        x = x_next


    return x_next


# 返回值K 即为LQR 问题求解方法中系数K的解

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation

    X = solve_dare(A, B, Q, R)

    # compute the LQR gain

    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)


    return K

