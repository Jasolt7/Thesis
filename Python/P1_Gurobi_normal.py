# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 22:47:28 2025

@author: JAPGc
"""

import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

pro_i = np.array([3, 5, 10, 1, 10])
pro_i = np.repeat(np.arange(len(pro_i)), pro_i)

n = pro_i.shape[0]  # Update `n` dynamically

rec_lpk = np.load('rec_lpk.npy')
rec_max_p = np.array([0, 0, 4, 4, 4, 2, 1, 2, 1, 4, 1, 2, 3, 1, 1, 2, 1, 6, 1, 3])

max_rows = rec_lpk.shape[1]
max_cols = rec_lpk.shape[2]
n_exams = rec_lpk.shape[0]
np.random.seed(50)
alpha = 1 
T = int(np.sum(rec_lpk[pro_i,1,:]))  # time horizon

pro_time = np.reshape(np.zeros(n_exams*max_cols),(n_exams,max_cols))

for j in range(n_exams):
    pro_time[j,0] = rec_lpk[j,1,0]
    for k in range(1,max_cols):
        pro_time[j,k] = rec_lpk[j,1,k] + pro_time[j,k-1]

threshold = rec_max_p[:].reshape(1, 1, -1)


model = Model(name="Sched")

np.random.seed(50)

# Variables
X_t_ik = model.addVars(T, n, max_cols, name='X_t_ik', vtype=GRB.BINARY)  # 1 if patient i is in surgery at time t during day g, 0 otherwise
Z_t_ik = model.addVars(T, n, max_cols, name='Z_t_ik', vtype=GRB.BINARY)  # 1 if patient i is in surgery at time t during day g, 0 otherwise

F_ik = model.addVars(n, max_cols, name="F_ik", vtype=GRB.INTEGER, lb=0, ub=T)             # End time of surgery i, day g
S_ik = model.addVars(n, max_cols, name="S_ik", vtype=GRB.INTEGER, lb=0, ub=T)             # Start time of surgery i, day g
M = model.addVar(name="M", vtype=GRB.INTEGER, lb=0, ub=T)

# Objective: Maximize total utilization
model.setObjective(M, GRB.MINIMIZE)

# Constraints

model.addConstrs(
    F_ik[i, max_cols-1] <= M for i in range(n)
)


model.addConstrs(
    quicksum(Z_t_ik[t, i, k] for t in range(T)) == 1
    for i in range(n) for k in range(max_cols)
)


# If a surgery is assigned, it must occupy its duration
"""
model.addConstrs(
    rec_lpk[pro_i[i], 1, k] == quicksum(X_t_ikg[t, i, k, g] for t in range(T))
    for i in range(n) for k in range(max_cols) for g in range(D)
)
"""
for i in range(n):
    for k in range(max_cols):
        duration = rec_lpk[pro_i[i], 1, k]
        model.addConstr(
            quicksum(X_t_ik[t, i, k] for t in range(T)) == duration
        )


for i in range(n):
    for k in range(max_cols):
        # Vincular Z ao bloco de X
        for t in range(T - rec_lpk[pro_i[i], 1, k]):
            model.addConstr(
                quicksum(X_t_ik[t_prime, i, k] for t_prime in range(t, t + rec_lpk[pro_i[i], 1, k])) >= rec_lpk[pro_i[i], 1, k] * Z_t_ik[t, i, k]
            )
                


model.addConstrs(
    S_ik[i, k+1] == F_ik[i, k]
    for i in range(n) for k in range(max_cols-1)
)

model.addConstrs(
    S_ik[i, k] == quicksum(t*Z_t_ik[t, i, k] for t in range(T))
    for i in range(n) for k in range(max_cols)
)

model.addConstrs(
    quicksum(rec_lpk[pro_i[i], p, k]*X_t_ik[t, i, k] for i in range(n) for k in range(max_cols)) <= rec_max_p[p]
    for t in range(T) for p in range(2, max_rows)
)


model.addConstrs(
    F_ik[i, k] - S_ik[i, k] == rec_lpk[pro_i[i], 1, k]
    for i in range(n) for k in range(max_cols)
)

# Solve
model.setParam('TimeLimit', 60*60*8)

model.setParam('Cuts', 3)  # Use aggressive cuts for faster convergence
model.setParam('Threads', 10)
model.setParam('NodeMethod', 2)

model.optimize()

"""
for t in range(T):
    for g in range(D):
        for p in range(2, max_rows):
            usage = sum(
                rec_lpk[pro_i[i], p, k] * X_t_ikg[t, i, k, g].X
                for i in range(n) for k in range(max_cols)
            )  # Compute the actual value first

            if usage == rec_max_p[p]:  # Now compare the computed value
                print(f"Resource {p} is fully used at time {t}.")
"""