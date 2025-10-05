# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 01:08:54 2025

@author: JAPGc
"""

import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def figplot(S_ik, F_ik, n, T, max_cols):
    
    # Plot a Gantt chart of scheduled tasks.
    fig, ax = plt.subplots(figsize=(40, 24))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    for i in range(n):
        for k in range(max_cols):
            start = S_ik[i, k]
            finish = F_ik[i, k]
            if finish > start and finish < T:
                ax.barh(y=i, width=finish - start, left=start,
                        height=0.8, color=colors[k % len(colors)], edgecolor='black')
                ax.text(start + (finish - start)/2, i,
                        f'B{k}', va='center', ha='center', color='white', fontsize=10, rotation=90)
    
    ax.set_xlabel("Tempo (min)", fontsize=18)
    ax.set_ylabel("Tarefa/Paciente", fontsize=18)
    ax.set_title("Agendamento - Gráfico de Gantt", fontsize=24)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Paciente {i}" for i in range(n)], fontsize=14)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_resource_utilization(rec_max_p, T, RU_temp, rec_lpk, pro_i):
    """
    Plot resource utilization using the precomputed resource usage matrix (RU_temp).
    
    Parameters:
        rec_max_p: 1D array of maximum resource capacities.
        T: Total time (int).
        RU_temp: 2D numpy array (shape: T x number_of_resources) holding resource usage over time.
        rec_lpk: (Unused here but preserved for interface consistency)
        pro_i: (Unused here but preserved for interface consistency)
    """
    recurso = ["tarefa", "duração", "RH TAS", "RH Enf", "RH Tec", "RH Med", 
               "RH Card", "Gab Med", "S2 Bal", "S2 Esp", "S Cam Gama", "S Cortina", 
               "S Esp Criança", "S Pol", "S Enf 1", "S Enf 1 ou S Pol", "S Enf 2", "S Enf 2 ou 3", 
               "S Tom", "WC" ]
    """
    rec4 = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8], # task             0
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # duration         0
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH TAS           4
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Enf           4
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Tec           4
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Med           2
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Card          1
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # Gab Med          2
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S2 Bal           1
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S2 Esp           4
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Cam Gama       1
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Cortina        2
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Esp Criança    3
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Pol            1
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1          1
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2          1
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 ou 3     6
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Tom            1
        [0, 0, 0, 0, 0, 0, 0, 0, 0]  # WC               3
        ])
    """
    
    # Filter out first two resource columns (i.e., "tarefa" and "duração")
    rec_max_p_filtered = rec_max_p[2:]
    recurso_filtered = recurso[2:]
    n_resources_filtered = len(rec_max_p_filtered)

    # Build the fully utilized matrix for filtered resources:
    # For each resource, mark a time instance as fully utilized if
    # the usage equals the maximum capacity.
    fully_utilized = np.zeros((n_resources_filtered, T), dtype=int)
    for p in range(n_resources_filtered):
        fully_utilized[p, :T] = (RU_temp[:T, p] == rec_max_p_filtered[p]).astype(int)

    # Plot the fully utilized status.
    plt.figure(figsize=(20, 10))
    plt.imshow(fully_utilized, aspect='auto', interpolation='none', cmap='Greys', origin='lower')
    plt.colorbar(label='Fully Utilized (1) vs Not (0)')
    plt.xlabel("Time (minutes)", fontsize=14)
    plt.ylabel("Resource", fontsize=14)
    plt.yticks(range(n_resources_filtered), recurso_filtered, fontsize=12)
    plt.title("Resource Full Utilization Over Time (Filtered)", fontsize=18)
    plt.show()

    # Compute the usage ratio for each resource over time.
    usage_ratio = RU_temp[:T].T / rec_max_p_filtered[:, np.newaxis]
    plt.figure(figsize=(20, 10))
    plt.imshow(usage_ratio, aspect='auto', interpolation='none', cmap='viridis', origin='lower')
    plt.colorbar(label='Usage Ratio (0 = no usage, 1 = fully utilized)')
    plt.xlabel("Time (minutes)", fontsize=14)
    plt.ylabel("Resource", fontsize=14)
    plt.yticks(np.arange(n_resources_filtered), recurso_filtered, fontsize=12)
    plt.title("Resource Utilization Over Time (Scaled by Usage)", fontsize=18)
    plt.show()

def precompute_exam_type_patterns(pro_i, X_t_ik, S_ik, F_ik, R):
    """
    Precompute exam patterns and their resource deltas by exam type.
    
    Parameters:
        pro_i: 1D array of exam types.
        X_t_ik: 3D scheduling array.
        S_ik, F_ik: 2D arrays of start and finish times.
        R: Resource consumption array corresponding to exams.
        
    Returns:
        exam_type_patterns: dict mapping exam type -> binary pattern array.
        exam_type_deltas: dict mapping exam type -> resource delta array.
    """
    exam_type_patterns = {}
    exam_type_deltas = {}
    
    for exam_type in np.unique(pro_i):
        # Get a representative exam of this type.
        i = np.where(pro_i == exam_type)[0][0]
        start = int(S_ik[i, 0])
        finish = int(F_ik[i, -1])
        pattern = X_t_ik[start:finish, i, :].copy()
        exam_type_patterns[exam_type] = pattern
        # Compute the resource usage delta for this exam type.
        delta = pattern @ R[i].T
        exam_type_deltas[exam_type] = delta
        
    return exam_type_patterns, exam_type_deltas

def ini_res_usage(X_t_ik, R, T):
    resource_usage = np.tensordot(X_t_ik[0:T, :, :], R, axes=([1, 2], [0, 2]))
    return resource_usage

pro_i = np.array([3, 5, 10, 1, 10])*2
pro_i = np.repeat(np.arange(len(pro_i)), pro_i)
selected_exams = np.array(['Linfocintigrafia para Detecção de gânglio sentinela', 'Cintigrafia Miocárdica de perfusão em esforço/stress farmacológico', 'Cintigrafia Miocárdica de perfusão em repouso'])

n = pro_i.shape[0]  # Update `n` dynamically

rec_lpk = np.load('rec_lpk.npy')
rec_max_p = np.array([0, 0, 4, 4, 4, 2, 1, 2, 1, 4, 1, 2, 3, 1, 1, 2, 1, 6, 1, 3])

max_rows = rec_lpk.shape[1]
max_cols = rec_lpk.shape[2]
n_exams = rec_lpk.shape[0]
np.random.seed(50)
T = int(487)  # time horizon

pro_time = np.reshape(np.zeros(n_exams*max_cols),(n_exams,max_cols))

for j in range(n_exams):
    pro_time[j,0] = rec_lpk[j,1,0]
    for k in range(1,max_cols):
        pro_time[j,k] = rec_lpk[j,1,k] + pro_time[j,k-1]

threshold = rec_max_p[:].reshape(1, 1, -1)
R = rec_lpk[pro_i, 2:, :]

X_t_ik = np.zeros((T, n, max_cols))
S_ik = np.zeros((n, max_cols))
F_ik = np.zeros((n, max_cols))
Y_i = np.zeros(n)

for i in range(n):
    t = np.random.randint(0, T - sum(rec_lpk[pro_i[i], 1]))
    S_ik[i, 0] = t
    F_ik[i, 0] = S_ik[i, 0] + rec_lpk[pro_i[i], 1, 0]
    Y_i[i] = 1

    for t_prime in range(int(S_ik[i, 0]), int(F_ik[i, 0])):
        X_t_ik[t_prime, i, 0] = 1
        
    for k in range(1, max_cols):
        S_ik[i, k] = F_ik[i, k-1]
        F_ik[i, k] = S_ik[i, k] + rec_lpk[pro_i[i], 1, k]
        for t_prime in range(int(S_ik[i, k]), int(F_ik[i, k])):
            X_t_ik[t_prime, i, k] = 1
    t = F_ik[i, max_cols-1]
    
exam_type_patterns, exam_type_deltas = precompute_exam_type_patterns(pro_i, X_t_ik, S_ik, F_ik, R)


model = Model(name="Sched")

np.random.seed(50)

# Decision vars
Z = model.addVars(n, T, vtype=GRB.BINARY, name="Z")  # start-time binaries
Y = model.addVars(n, vtype=GRB.BINARY, name="Y")     # exam selected

# Link Z to Y
for i in range(n):
    e     = pro_i[i]
    delta = exam_type_deltas[e]    # shape (dur_i, max_rows-2)
    dur   = delta.shape[0]
    model.addConstr(
        quicksum(Z[i, t] for t in range(T-dur+1)) == Y[i])

# Objective: maximize sum(Y)
model.setObjective(Y.sum(), GRB.MAXIMIZE)

# Resource constraints
for p_idx in range(exam_type_deltas[0].shape[1]):
    cap = rec_max_p[p_idx+2]   # remember delta’s p_idx=0 corresponds to original p=2
    for t in range(T):
        expr = LinExpr()
        for i in range(n):
            e     = pro_i[i]
            delta = exam_type_deltas[e]    # shape (dur_i, max_rows-2)
            dur   = delta.shape[0]
            # only starts τ that cover time t
            τ_min = max(0, t - dur + 1)
            τ_max = min(t + 1, T - dur + 1)
            for τ in range(τ_min, τ_max):
                usage = delta[t - τ, p_idx]
                if usage:
                    expr.add(Z[i, τ], usage)
        model.addConstr(expr <= cap)
        

# Solve
model.setParam('TimeLimit', 60*60*8)
model.setParam('Cuts', 3)  # Use aggressive cuts for faster convergence
model.setParam('Threads', 10)

model.optimize()

# After solving first model
Y_i = np.zeros(n, dtype=int)
S_ik = np.zeros((n, max_cols))

# Populate from Z solution
for i in range(n):
    for t in range(T):
        if Z[i, t].X > 0.5:  # tolerance for float precision
            Y_i[i] = 1
            S_ik[i, 0] = t  # first stage start time

# Now build F_ik and X_t_ik from these start times
F_ik = np.zeros((n, max_cols))
X_t_ik = np.zeros((T, n, max_cols))

for i in range(n):
    if Y_i[i] == 1:
        # Stage 0
        F_ik[i, 0] = S_ik[i, 0] + rec_lpk[pro_i[i], 1, 0]
        for t_prime in range(int(S_ik[i, 0]), int(F_ik[i, 0])):
            X_t_ik[t_prime, i, 0] = 1

        # Remaining stages
        for k in range(1, max_cols):
            S_ik[i, k] = F_ik[i, k - 1]
            F_ik[i, k] = S_ik[i, k] + rec_lpk[pro_i[i], 1, k]
            for t_prime in range(int(S_ik[i, k]), int(F_ik[i, k])):
                X_t_ik[t_prime, i, k] = 1
                
resource_usage = ini_res_usage(X_t_ik, R, T)
exam_type_patterns, exam_type_deltas = precompute_exam_type_patterns(pro_i, X_t_ik, S_ik, F_ik, R)

figplot(S_ik, F_ik, n, T, max_cols)
plot_resource_utilization(rec_max_p, T, resource_usage, rec_lpk, pro_i)