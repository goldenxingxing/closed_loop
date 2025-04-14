# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:11:01 2025

@author: liqunwei
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
from math import floor
from tqdm import tqdm
import cvxpy as cp

def solve_DARE( A, C, Q, R, maxiter=1500, eps=0.01):
    """
    Solve a discrete time_Algebraic Riccati equation (DARE)
    I tested it and found the iteration ends with 30 steps
    """
    P = A

    for i in range(maxiter):
        INV = R + C @ P @ C.T
        Pn = A @ P @ A.T  + Q - A @ P @ C.T @  C @ P @ A.T / INV[0,0]
        # Pn = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if (abs(Pn - P)).max() < eps:
            break
        P = Pn
        # print(i)

    K = -np.linalg.inv(C @ P @ C.T+R) @ C @ P @ A.T
    L = K.T
    print('solve_DARE uses {} steps'.format(i))
    return L



class system_model:
    def __init__(self,TDI, N, G, I, X, Q, R,timescale):
        self.meal = True
        self.velocity = False
        self.I = I
        self.G = G
        self.G_pred = G
        self.p1 = 0.98
        self.p2 = 0.965
        self.A = np.array([[2.91,-2.8226,0.9126],[1,0,0],[0,1,0]])
        self.B = np.array([[-1800*90*(1-self.p1)*(1-self.p2)**2/TDI],[0],[0]])
        self.C = np.array([[0,0,1]])
        self.Q = Q
        self.v = 0
        self.Cv = np.array([[0.1,0,-0.1]])


        self.L = solve_DARE(A=self.A, C=self.C, Q=Q, R=R)
        self.X = X
        self.R = R
        self.N = N
        self.TDI = TDI
        self.basal = TDI/24/60
        self.C_f = 40 # deepseek gives 30-50
        self.G_ref = 110
        self.timescale = timescale # min
        self.I_record = np.zeros(int(8*60/self.timescale))
#I is insulin delivery (pmol/min), G is BG concentration (mg/dL)
# 1 pmol of insulin is 1.67e-4 U
    def theta_IOB(self, theta):
        # insulin 75 min at peak
        # theta = theta * 60 / self.timescale
        duration = int(8*60/self.timescale)
        iob_curve = np.zeros(duration)
        for i in range(duration):
            t = i /duration * 8
            if  t < theta/2:
                r = 1-2*max(0,t)**2/theta**2
            else:
                t = min(theta,t)
                r = 2*(theta-t)**2/theta**2
            iob_curve[i] = r
        return iob_curve

    def IOB_upper_bound(self):
        TAU = self.G/self.C_f
        g_cgm = self.G + self.G_ref
        if g_cgm > 300:
            theta = 2
        elif g_cgm > 200:
            theta = 4
        elif g_cgm > 140:
            theta = 6
        else:
            theta = 8

        theta = self.theta_IOB(theta)
        THETA = np.dot(theta, self.I_record+self.basal)
        return max(TAU-THETA, 0)




    def update_next(self, i):
        self.X = self.A @ self.X + self.B * i - self.L * (self.G - self.G_pred)
        G_pred = self.C @ self.X
        self.G_pred = G_pred[0,0]
        self.G = self.G_pred
        self.I_record = np.append(self.I_record, i)[1:]
        v = self.Cv @ self.X
        self.v = v[0,0]



    # def update_X(self, i):
    #     # , xi = [yi+2 yi+1 yi]T
    #     # self.X[2,0] = self.G
    #     self.X = self.A @ self.X + self.B * i - self.L * (self.G - self.G_pred)
    #     # self.X[2,0] = self.G
    #     G_pred = self.C @ self.X
    #     self.G_pred = G_pred[0,0]
    #     self.G = self.G_pred
    #     self.I_record = np.append(self.I_record, i)[1:]

    def mat_N(self):
        ini = np.zeros((3,self.N))
        ini = np.append(self.A,ini,axis=-1)
        row = np.eye(3)
        system_mat = np.empty((0,3+self.N))
        for i in range(self.N):
            row = self.A @ row
            row = np.append(row,self.B,axis=-1)
            zeros = np.zeros((3,self.N-1-i))
            row_i = np.append(row, zeros, axis=-1)
            row_i = self.C@row_i
            system_mat = np.concatenate((system_mat,row_i) ,axis=0)
        self.system_mat = system_mat

    def predict_N_mat(self,I):
        self.mat_N()
        I = I.reshape(self.N,1)
        state_mat = np.concatenate((self.X, I),axis=0)
        return (self.system_mat@state_mat).flatten()

    def predict_N_mat_cvx(self,I):
        self.mat_N()
        I = I.reshape(self.N,1)
        state_mat = np.concatenate((self.X, I),axis=0)
        return (self.system_mat@state_mat).flatten()

    def predict_N(self,I):
        X = self.X
        G = self.G
        G_pred = self.G_pred
        # G_N = np.empty((1,0))
        G_N = np.zeros(self.N)
        for i in range(self.N):
            p1 = self.A @ X

            p2 = self.B * I[i]
            p3 = self.L * (G - G_pred)
            X = p1 + p2 + p3
            G_pred = self.C @ X #(1,1)
            G_pred = G_pred[0,0]
            G = G_pred
            G_N[i] = G_pred
        return G_N

def mpc_cost(I, system_model, R, Q, G_ref=0, I_basal=1/6):
    G_N = system_model.predict_N(I)
    result = np.sum(np.square(G_N-G_ref))*Q + np.sum(np.square(I))*R
    IOB = np.sum(np.square(I-system.IOB_upper_bound()))*R
    return result + IOB

def mse_cost(I, system_model, R, Q, G_ref=0, I_basal=1/6):
    G_N = system_model.predict_N(I)
    result = np.sum(np.square(G_N-G_ref))*Q + np.sum(np.square(I))*R

    return result

def solve_mpc(I_ini,N,system_model, Q, R):
    # Linear constraints on the rate of change of I: -delta_I_max <= I[i+1] - I[i] <= delta_I_max for all i in the N - 1
    # Implemented using LinearConstraint as -delta_I_max <= delta_I_matrix * I <= delta_I_max
    delta_I_max = 0.5
    I_max = 0.5
    delta_I_matrix = np.eye(N) - np.eye(N, k=1)
    constraint1 = LinearConstraint(delta_I_matrix, -delta_I_max*np.ones(N), delta_I_max*np.ones(N))

    # # # We need a constraint on the rate of change of I[0] respect to its previous value, which is I_ini[0]
    first_I_matrix = np.zeros([N, N])
    first_I_matrix[0, 0] = 1
    constraint2 = LinearConstraint(first_I_matrix, (I_ini[0]-delta_I_max)*np.ones(N), (I_ini[0]+delta_I_max)*np.ones(N))

    G_t_constraint = NonlinearConstraint(system_model.predict_N_mat, -40, 50)
    # IOB_constraint = NonlinearConstraint(system_model.IOB_upper_bound, -I_max, 0 )

    # discrete_constraint = NonlinearConstraint(system_model.discrete_I, 0.0, 0.0)

    # Add constraints
    # constraint = [constraint1,constraint2,G_t_constraint, IOB_constraint]

    constraint = [constraint1,constraint2,G_t_constraint]

    bounds = [(-system_model.TDI/24/60*system_model.timescale, I_max) for idx in range(N)]

    # Minimization
    result = minimize(mse_cost, I_ini, args=(system_model, R, Q), bounds=bounds,  constraints=constraint)
    # result = minimize(mpc_cost, I_ini, args=(system_model, R, Q), bounds=bounds)
    # Extract the optimal control sequence
    I_mpc = result.x

    return I_mpc, system_model.X, result

def solve_cvx_mpc(N,system_model, Q, R):
    g = cp.Variable((N,1))
    i = cp.Variable((N,1))
    constraints = []
    # A, B, C, L = system_model.A, system_model.B, system_model.C, system_model.L
    # X, G, G_pred = system_model.X, system_model.G, system_model.G_pred
    cost = 0
    system_model.mat_N()
    # I = I.reshape(self.N,1)
    # state_mat = np.concatenate((self.X, I),axis=0)
    big_mat =system_model.system_mat
    big_mat_x, big_mat_i = np.array_split(big_mat, [3], axis=1)

    G_pred = big_mat_x@system_model.X
    G_pred = G_pred+big_mat_i@i

    constraints = [g==G_pred]
    # cost += cp.quad_form(g[:,0], Q)+cp.quad_form(i[:,0], R)
    cost += Q*cp.norm(g, p=2)**2 + R*cp.norm(i, p=2)**2
    constraints += [i >= -system_model.TDI/24/60*system_model.timescale/2]
    prob = cp.Problem(cp.Minimize(cost), constraints)
   # 求解最小二乘
    # prob.solve(solver=cp.SCS)
    prob.solve(solver=cp.SCS)
    return i.value[0]

HOURS = 24 #7am to 9pm
timescale = 5
N_iterations = int(floor(HOURS*60/timescale))

# Arrays for logging
I = np.zeros(N_iterations)
# ---------- CONTROL SYSTEM CALIBRATION ----------
# Model predictive control horizon
N = 30

# Array for values of theta and I for each simulation
G_ini = 0
TDI = 40
I_ini = np.ones(N)*TDI/24/60*timescale
X_ini =np.array([[1],[1],[1]])*G_ini
Q = 1
R = 1500
system = system_model(TDI, N, G_ini, I_ini, X_ini, Q, R,timescale)

bfast_start = int(floor(2*60/timescale)) #
bfast_end = int(floor(2.5*60/timescale))
lunch_start = int(floor(5*60/timescale))
lunch_end = int(floor(6*60/timescale))
dinner_start = int(floor(7.5*60/timescale))
dinner_end = int(floor(8*60/timescale))

G = np.zeros(N_iterations)
L_plasma = 45 # dl
# system.meal=True
velocity = True
for idx in tqdm (range(N_iterations), desc="Loading..."):
    # I_mpc,x,result = solve_mpc(I_ini,N,system, Q, R)
    I_mpc = solve_cvx_mpc(N, system, Q, R)
    # Initial solution for next step = current solution
    I[idx] = I_mpc[0]
    if idx in (bfast_start, lunch_start, dinner_start):
        I[idx] += 0
    # ---------- SIMULATION LOOP  ----------
    system.update_next(I_mpc[0])
    if velocity:
        v = system.v
        epsilon = 1e-6
        if v>=0:
            Q = 1
        elif v<=-1:
            Q = epsilon
        else:
            Q = 0.5*(np.cos(v*np.pi)*(1-epsilon)+(1-epsilon))
    I_ini = I_mpc
    if system.meal:
        if (idx>=bfast_start) & (idx<=bfast_end): system.G += (1/(bfast_end-bfast_start))*50/L_plasma*1000 # 300 mg/dl of glucose infused of 15 mins
        if (idx>=lunch_start) & (idx<=lunch_end): system.G += (1/(lunch_end-lunch_start))*50/L_plasma*1000 #
        if (idx>=dinner_start) & (idx<=dinner_end): system.G += (1/(dinner_end-dinner_start))*50/L_plasma*1000

    G[idx] = system.G


I += TDI/24/60*timescale

x=np.arange(N_iterations)*timescale/60
plt.subplot(2, 1, 1)
plt.plot(x, G+110, '--', label="Glucose Level")
plt.ylabel(r"G_t, mg/dL")
plt.axhline(70.,color='red',linestyle='--',linewidth=0.75)
plt.axhline(140.,color='green',linestyle='--',linewidth=0.75)
if system.meal:
    plt.axvline(bfast_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_start*timescale/60,color='red',linestyle='--',linewidth=0.75)

    plt.axvline(bfast_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)


plt.subplot(2, 1, 2)

plt.plot(x, I, 'o', label="Insulin",marker = "o",markersize=0.5, mec = 'b', mfc = 'b')
plt.ylabel(r"I_u, U")
plt.xlabel("Time")
if system.meal:
    plt.axvline(bfast_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_start*timescale/60,color='red',linestyle='--',linewidth=0.75)

    plt.axvline(bfast_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
plt.show()