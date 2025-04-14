# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:33:39 2025

@author: liqunwei
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
from math import floor

#based on alternative single explicti delay model Eq(10) in https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7fdec5a9d98696d5533d773c7ce9018e7fddaabd

class system_model:
    def __init__(self, N, G, I):
        self.meal = True
        self.I = I
        self.G = G
        self.G_pred = G
        self.N = N

        self.G_ref = 110
        self.timescale = 5 # min

        self.x1 = 0
        self.x2 = 0
        self.x3 = 0

        self.td = 36

        self.Vg = 10
        self.Vp = 3
        self.Vi = 11
#I is insulin delivery (pmol/min), G is BG concentration (mg/dL)
# 1 pmol of insulin is 1.67e-4 U
    def f2(self,G):
        Ub = 72
        C2 = 144
        return Ub*(1-np.exp(-G/(C2*self.Vg)))
    def f3(self,G):
        C3 = 1000
        return G/(C3*self.Vg)
    def f4(self,I):
        U0 = 40
        # Um = 940 *self.time_scale #scaled by 5 to convert to 5 minute time increments
        Um = 940
        beta = 1.77
        C4 = 80
        # E = 0.2 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        E = 0.2
        # ti = 100 /self.time_scale # scaled by 1/5 to convert to 5 minute time increments
        ti = 100
        return U0 + (Um-U0)/(1+np.exp(-beta*np.log(0.0000001+I/C4*(1/self.Vi+1/(E*ti)))))
    def f5(self,I):
        # Rg = 180 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        Rg = 180
        alpha = 0.29
        C5 = 26
        x = alpha*(I/self.Vp - C5)
        if x>=0:
            y = Rg*np.exp(-x) / (np.exp(-x) + 1)
        else:
            y = Rg/(1+np.exp(x))
        return y

    def update_x(self):
        d_x1 = 3*(self.I-self.x1)/self.td
        d_x2 = 3*(self.x1-self.x2)/self.td
        d_x3 = 3*(self.x2-self.x3)/self.td

        self.x1 += d_x1*self.timescale
        self.x2 += d_x2*self.timescale
        self.x3 += d_x3*self.timescale

    def update_next(self, i):
        dG = -self.f2(self.G)-self.f3(self.G)*self.f4(self.I) + self.f5(self.x3)
        dI = i -self.timescale*0.06*self.I
        self.update_x()
        self.I += dI
        self.G += self.timescale*dG

    def predict_N(self,tau):
        x1 = self.x1
        x2 = self.x2
        x3 = self.x3
        G = self.G
        I = self.I
        G_N = np.zeros(self.N)
        for i in range(self.N):
            dG = -self.f2(G)-self.f3(G)*self.f4(I) + self.f5(x3)
            dI = tau[i] -self.timescale*0.06*I
            d_x1 = 3*(I-x1)/self.td
            d_x2 = 3*(x1-x2)/self.td
            d_x3 = 3*(x2-x3)/self.td

            x1 += d_x1*self.timescale
            x2 += d_x2*self.timescale
            x3 += d_x3*self.timescale
            I += dI
            G += self.timescale*dG
            G_N[i] = G
        return G_N

def mpc_cost(tau, system_model,g, G_t_ref=110):
    G_t_N = system_model.predict_N(tau)
    return np.sum(np.square(np.array(G_t_N)-G_t_ref*g))

def solve_mpc(tau_ini,N,system_model,g):
    delta_tau_max = 1000
    tau_max = 1500
    delta_tau_matrix = np.eye(N) - np.eye(N, k=1)
    constraint1 = LinearConstraint(delta_tau_matrix, -delta_tau_max, delta_tau_max)

    # # # We need a constraint on the rate of change of tau[0] respect to its previous value, which is tau_ini[0]
    first_element_matrix = np.zeros([N, N])
    first_element_matrix[0, 0] = 1
    constraint2 = LinearConstraint(first_element_matrix, 0, tau_ini[0]+delta_tau_max)
    G_t_constraint = NonlinearConstraint(system_model.predict_N, 50*g, 200*g)
    # discrete_constraint = NonlinearConstraint(system_model.discrete_tau, 0.0, 0.0)

    # Add constraints
    delta_tau_constraint = [constraint1,constraint2,G_t_constraint]

    bounds = [(0, tau_max) for idx in range(N)]

    # Minimization

    # result = minimize(m_cost, tau_ini, args=(system_model, g), bounds=bounds, constraints=delta_tau_constraint)

    result = minimize(mpc_cost, tau_ini, args=(system_model, g), bounds=bounds)
    # Extract the optimal control sequence
    tau_mpc = result.x

    return tau_mpc

HOURS = 24 #7am to 9pm
timescale = 5
N_iterations = int(floor(HOURS*60/timescale))

# Arrays for logging
I = np.zeros(N_iterations)
# ---------- CONTROL SYSTEM CALIBRATION ----------
# Model predictive control horizon
N = 15
tau_init = np.ones(N)*0.1
tau = np.zeros(N_iterations)
# Array for values of theta and I for each simulation
g=100
G_ini = 110*g

system = system_model(N, G_ini, I=0.5)

bfast_start = int(floor(2*60/timescale)) #
bfast_end = int(floor(2.5*60/timescale))
lunch_start = int(floor(5*60/timescale))
lunch_end = int(floor(6*60/timescale))
dinner_start = int(floor(7.5*60/timescale))
dinner_end = int(floor(8*60/timescale))

G_t = np.zeros(N_iterations)

# system.meal=True
for idx in range(N_iterations):
    tau_mpc = solve_mpc(tau_init,N,system,g)
    # Initial solution for next step = current solution
    tau[idx] = tau_mpc[0]
    # ---------- SIMULATION LOOP  ----------
    system.update_next(tau_mpc[0])
    tau_init = tau_mpc
    if system.meal:
        if (idx>=bfast_start) & (idx<=bfast_end): system.G += (1/(bfast_end-bfast_start))*50*1000 # 300 mg/dl of glucose infused of 15 mins
        if (idx>=lunch_start) & (idx<=lunch_end): system.G += (1/(lunch_end-lunch_start))*50*1000 #
        if (idx>=dinner_start) & (idx<=dinner_end): system.G += (1/(dinner_end-dinner_start))*50*1000
    G_t[idx] = system.G

G_predicted = G_t
# x=np.arange(N_iterations)*timescale/60
# plt.subplot(2, 1, 1)
# plt.plot(x, G/g, '--', label="Glucose Level")
# plt.subplot(2, 1, 2)
# plt.plot(x, I, '--', label="Insulin")
# plt.show()

with open('plot.py', 'r') as file:
    code = file.read()
    exec(code)