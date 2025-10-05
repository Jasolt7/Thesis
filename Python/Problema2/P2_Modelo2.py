# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 23:05:09 2025

@author: JAPGc
"""

import numpy as np
from numpy.random import SeedSequence
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from numba import njit
import cProfile
import pstats
from line_profiler import LineProfiler

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

def deep_copy(*arrays):
    # Returns copies of all passed arrays.
    return [a.copy() for a in arrays]

def ini_res_usage(X_t_ik, R, T):
    resource_usage = np.tensordot(X_t_ik[0:T, :, :], R, axes=([1, 2], [0, 2]))

    return resource_usage

@njit
def vec_vio(resource_usage, start, end, threshold):
    n_rows = threshold.shape[1]
    threshold = threshold[0]
    count = 0
    for t in range(start, end):
        for j in range(n_rows):
            if resource_usage[t, j] > threshold[j]:
                count += 1
    return count

@njit
def compute_makespan(F_ik, T, pro_i, durations):
    start = T
    end = 0
    for i in range(F_ik.shape[0]):
        F = F_ik[i]
        dur = durations[i]
        if F - dur < start:
            start = F - dur
        if F > end:
            end = F
    return start, end

@njit
def shift_resource_usage_numba(resource_usage, delta, original_start, shift):
    n_rows = delta.shape[1]
    T = delta.shape[0]
    new_start = original_start + shift
    for t in range(T):
        for j in range(n_rows):
            val = delta[t, j]
            resource_usage[original_start + t, j] -= val
            resource_usage[new_start +t, j] += val 
    return resource_usage

@njit
def counter_numba(resource_usage, delta, threshold, t):
    duration = delta.shape[0]
    n_rows = threshold.shape[1]
    for i in range(duration):
        for j in range(n_rows):
            if resource_usage[t + i, j] + delta[i, j] > threshold[0, j]:
                return 1
    return 0

@njit
def earliest_candidate(resource_usage, delta, threshold, start, end, pri1, pri2, i, exam_type_start):
    duration = delta.shape[0]
    n_rows = threshold.shape[1]
    # Full search, but check priority resources first (usually much cheaper)
    start = max(start, exam_type_start[i])
    for t in range(start, end - duration + 1):
        valid = True
        # check priority resource pri1 and pri2 first
        for i in range(duration):
            v1 = delta[i, pri1]
            if v1 != 0.0:
                if resource_usage[t + i, pri1] + v1 > threshold[0, pri1]:
                    valid = False
                    break
            v2 = delta[i, pri2]
            if v2 != 0.0:
                if resource_usage[t + i, pri2] + v2 > threshold[0, pri2]:
                    valid = False
                    break
        if not valid:
            continue

        # check other resources
        for i in range(duration):
            for j in range(n_rows):
                if j == pri1 or j == pri2:
                    continue
                if resource_usage[t + i, j] + delta[i, j] > threshold[0, j]:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return t
    return -1

@njit
def get_start_end(Sequence, starting_time, pro_i, rec_lpk, n, max_cols):
    S_ik = np.zeros((n, max_cols))
    F_ik = np.zeros((n, max_cols))
    for i in range(len(Sequence)):
        seq_i = Sequence[i]
        S_ik[seq_i, 0] = starting_time[i]
        F_ik[seq_i, 0] = S_ik[seq_i, 0] + rec_lpk[pro_i[seq_i], 1, 0]
        for k in range(1, max_cols):
            S_ik[seq_i, k] = F_ik[seq_i, k-1]
            F_ik[seq_i, k] = S_ik[seq_i, k] + rec_lpk[pro_i[seq_i], 1, k]
            
    return S_ik, F_ik

def define_sequence_LS(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, start, end):
    Sequence = np.asarray(Sequence).ravel().astype(int)  # sanitize
    resource_usage = np.zeros((T, rec_lpk.shape[1]-2))
    starting_time = []
    n_jobs = n
    n = len(Sequence)
    pri_s_cam_gama = 8
    pri_s_tom = 16
    exam_type_start = [0]*np.unique(pro_i)
    
    for i in Sequence:
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        duration_glob = sum(rec_lpk[pro_i[i],1])
        if exam_type_start[exam_type] <= end:
            t = earliest_candidate(resource_usage, delta, threshold, start, end, pri_s_cam_gama, pri_s_tom, exam_type, exam_type_start)
            if t == -1:
                t = T - duration_glob - 1

            if t > exam_type_start[exam_type]:
                exam_type_start[exam_type] = t
        else:
            t = T - duration_glob - 1
        starting_time.append(t)
        resource_usage[t : t + duration_glob, :] += delta

    S_ik, F_ik = get_start_end(Sequence, starting_time, pro_i, rec_lpk, n_jobs, max_cols)
    
    makespan = -np.count_nonzero(F_ik[:, -1] <= end)
    
    return S_ik, F_ik, resource_usage, makespan

def initial_solution(threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas):
    order = np.zeros(n)
    for i in range(n):
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        order[i] = delta.shape[0]
        
    order = np.flip(np.argsort(order))
    Sequence = order[0:2]   # start with first two
    
    for step in range(2, len(pro_i)):
        Sequence_size = len(Sequence) + 1
        best_cost = float("inf")
        best_seq = None
        
        for l in range(Sequence_size):
            Sequence_alt = np.insert(Sequence, l, order[step])  # candidate
            _, F_alt, _, _ = define_sequence_LS(Sequence_alt, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
            alt_cost = -np.count_nonzero(F_alt[:, -1] <= T)
            if alt_cost < best_cost:
                best_cost = alt_cost
                best_seq = Sequence_alt.copy()
        
        Sequence = best_seq  # update only once, after testing all insertions
    
    return Sequence

@njit(fastmath=True)
def valid_candidates_numba_func(resource_usage, delta, threshold, makespan):
    valid_candidates = []
    duration = delta.shape[0]
    n_rows = threshold.shape[1]
    threshold_flat = threshold.ravel()  # Pre-flatten
    count = 0
    for t in np.arange(0, makespan - duration):
        for i in range(duration):
            t_offset = t + i
            for j in range(n_rows):
                count += (resource_usage[t_offset, j] + delta[i, j]) > threshold_flat[j]

            if count > 0:
                break

        if count > 0:
            count = 0
        else:
            valid_candidates.append(t)

    return valid_candidates

@njit
def get_size(Y_i, idx):
    count = 0
    for i in range(len(Y_i)):
        if Y_i[i] == idx:
            count +=1
    return count

@njit
def get_array(Y_i, idx):
    array = []
    for i in range(len(Y_i)):
        if Y_i[i] == idx:
            array.append(i)
    return array

def sa_iteration(Y_i, S_ik, F_ik, resource_usage, c, pro_time, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish, nei_1, nei_2):
    F_temp, S_temp, Y_temp, RU_temp = deep_copy(F_ik, S_ik, Y_i, resource_usage)
    F_best, S_best, Y_best, RU_best = deep_copy(F_ik, S_ik, Y_i, resource_usage)
        
    vec_violations = np.count_nonzero(RU_temp > threshold)
    current_cost = -np.where(Y_temp == 1)[0].size + punish * vec_violations
    best_cost = current_cost
    new_start = 0
    new_finish = T
    
    if vec_violations == 0:
        violation_free = True
    else:
        violation_free = False
    
    accepted = 0
    n = pro_i.shape[0]
    max_cols = R.shape[2]
    
    i_it = np.random.rand(Lk)
    q_it = np.random.rand(Lk)
    shift_it = np.random.rand(Lk)
    neighbours = np.random.rand(Lk)
    nei_0 = 1 - nei_1 - nei_2
    average_dif = 0

    for it in range(Lk):
        if neighbours[it] < nei_0:
            neighbour = 0
        elif neighbours[it] < nei_0 + nei_1:
            neighbour = 1
        else:
            neighbour = 2
        
        #shift
        if neighbour == 0:
            assigned = np.where(Y_temp == 1)[0]
            if assigned.size == 0:
                neighbour = -1
                continue
            i = assigned[int(assigned.size * i_it[it])]
            
            exam_type = pro_i[i]
            delta = exam_type_deltas[exam_type]
            
            original_finish = int(F_temp[i, -1])
            original_start = original_finish - delta.shape[0]
        
            RU_temp[original_start:original_finish, :] -= delta
            duration = delta.shape[0]
            
            valid_candidates = np.array(valid_candidates_numba_func(RU_temp, delta, threshold, T))
            
            if valid_candidates.size > 0:
                start = np.random.choice(valid_candidates)
            else:
                start = original_start
            shift = start - original_start
            
            # Update start and finish times.
            S_temp[i, :] += shift
            F_temp[i, :] += shift
            
            # Insert the exam pattern into the new location.
            new_start = original_start + shift
            new_finish = original_finish + shift
            RU_temp[new_start:new_finish, :] += delta
        
        #remove
        elif neighbour == 1:
            assigned = np.where(Y_temp == 1)[0]
            if assigned.size == 0:
                neighbour = -1
                continue
            i = assigned[int(assigned.size * i_it[it])]
            original_start = int(S_temp[i, 0])
            original_finish = int(F_temp[i, -1])
            
            exam_type = pro_i[i]
            delta = exam_type_deltas[exam_type]
            
            RU_temp[original_start:original_finish, :] -= delta
            
            S_temp[i] = -1
            F_temp[i] = -1
            Y_temp[i] = 0
            
        #add  
        elif neighbour == 2:
            unassigned = np.where(Y_temp == 0)[0]
            if unassigned.size == 0:
                neighbour = -1
                continue
            i = unassigned[int(unassigned.size * i_it[it])]
            duration = int(np.sum(rec_lpk[pro_i[i],1,:]))
            exam_type = pro_i[i]
            delta = exam_type_deltas[exam_type]
            
            valid_candidates = np.array(valid_candidates_numba_func(RU_temp, delta, threshold, T))
        
            if valid_candidates.size > 0:
                start = np.random.choice(valid_candidates)
                F_temp[i, :] = start + pro_time[pro_i[i]]
                S_temp[i, :] = F_temp[i, :] - rec_lpk[pro_i[i], 1, :]
                original_start = int(S_temp[i, 0])
                original_finish = int(F_temp[i, -1])
                RU_temp[original_start:original_finish, :] += delta
                Y_temp[i] = 1
                
            else:
                neighbour = -1
                continue
            
        vec_violations = vec_vio(RU_temp, 0, T, threshold)
        alt_cost = -get_size(Y_temp, 1) + punish * vec_violations
        
        average_dif = average_dif + np.abs((current_cost - alt_cost))

        
        # Acceptance criterion of the SA algorithm.
        if alt_cost < current_cost or q_it[it] < 2.718281828459045**((current_cost - alt_cost) / c):
            current_cost = alt_cost
            if current_cost < best_cost:
                best_cost = current_cost
                RU_best = RU_temp.copy()
                S_best = S_temp.copy()
                F_best = F_temp.copy()
                Y_best = Y_temp.copy()
            accepted += 1

        else:
            if neighbour == 0:
                RU_temp[new_start:new_finish, :] -= delta
                S_temp[i, :] -= shift
                F_temp[i, :] -= shift
                RU_temp[original_start:original_finish, :] += delta
            
            elif neighbour == 1:
                RU_temp[original_start:original_finish, :] += delta
                
                start = original_start
                F_temp[i, :] = start + pro_time[pro_i[i]]
                S_temp[i, :] = F_temp[i, :] - rec_lpk[pro_i[i], 1, :]
                
                Y_temp[i] = 1
                
            elif neighbour == 2:
                RU_temp[original_start:original_finish, :] -= delta
                
                F_temp[i] = -1
                S_temp[i] = -1
                
                Y_temp[i] = 0

    return S_temp, F_temp, Y_temp, current_cost, accepted, RU_temp, S_best, F_best, Y_best, RU_best, best_cost, average_dif

def initial_temp(pro_time, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, punish, ratio, nei_1, nei_2):
    np.random.seed(seed)
    
    Sequence = initial_solution(threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas)
    S_ik, F_ik, resource_usage, makespan = define_sequence_LS(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)    
    Y_i = np.where(np.any(S_ik >= T-1, axis=1) | np.any(F_ik >= T-1, axis=1), 0, 1)

    # Step 2: mask S_ik and F_ik where Y_i == 0
    S_ik = np.where(Y_i[:, None] == 0, -1, S_ik)
    F_ik = np.where(Y_i[:, None] == 0, -1, F_ik)
    
    Lk = 5000
    _, _, _, _, acc, _, _, _, _, _, _, average_dif = sa_iteration(Y_i, S_ik, F_ik, resource_usage, float("inf"), pro_time, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish, 0.5, 0.5)
    c_in = np.abs((average_dif/Lk)/np.log(ratio))
    return c_in

def SA(pro_time, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, c_in, seed, Lk, freeze_crit, temp_red, punish, ratio, nei_1, nei_2):
    np.random.seed(seed)

    start_time = time.time()

    c = c_in
    best_cost = float('inf')

    resource_usage = np.zeros((T, rec_lpk.shape[1]-2))    
    S_ik = np.zeros((n, max_cols))
    F_ik = np.zeros((n, max_cols))
    Y_i = np.zeros(n)
    
    _, _, _, _, _, _, S_ik, F_ik, Y_i, resource_usage, _, _ = sa_iteration(Y_i, S_ik, F_ik, resource_usage, c_in, pro_time, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish, 0.5, 0.5)

    freeze = 0

    while freeze < freeze_crit:

        S_new, F_new, Y_new, cost_new, accepted, RU_new, S_best, F_best, Y_best, RU_best, best_cost_new, _ = sa_iteration(Y_i, S_ik, F_ik, resource_usage, c, pro_time, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish, nei_1, nei_2)
        if best_cost_new < best_cost:
            bestS, bestF, bestY, bestRU = deep_copy(S_best, F_best, Y_best, RU_best)
            best_cost = best_cost_new
            freeze = 0
        else:
            freeze += 1
        S_ik, F_ik, Y_i, resource_usage = deep_copy(S_new, F_new, Y_new, RU_new)

        c *= temp_red
        if c < 9.608478221092565e-100:
            c = 9.608478221092565e-100

    end_time = time.time() - start_time

    return bestS, bestF, bestY, c, bestRU, end_time
    
def main(n_runs, n_threads, Lk, freeze_crit, temp_red, punish, ratio, nei_1, nei_2, final_time, pro_i):        
    seed = 50
    np.random.seed(seed)  # Global seed
        
    pro_i = np.repeat(np.arange(len(pro_i)), pro_i)
    
    rec_lpk = np.load('rec_lpk.npy')
    rec_max_p = np.array([0, 0, 4, 4, 4, 2, 1, 2, 1, 4, 1, 2, 3, 1, 1, 2, 1, 6, 1, 3])

    max_rows = rec_lpk.shape[1]
    max_cols = rec_lpk.shape[2]
    n_exams = rec_lpk.shape[0]
    n = pro_i.shape[0]

    T = final_time
    R = rec_lpk[pro_i, 2:, :]
    threshold = rec_max_p[2:].reshape(1, -1)
    
    pro_time = np.reshape(np.zeros(n_exams*max_cols),(n_exams,max_cols))
    for j in range(n_exams):
        pro_time[j,0] = rec_lpk[j,1,0]
        for k in range(1,max_cols):
            pro_time[j,k] = rec_lpk[j,1,k] + pro_time[j,k-1]

    X_t_ik = np.zeros((T, n, max_cols))
    S_ik = np.zeros((n, max_cols))
    F_ik = np.zeros((n, max_cols))
    Y_i = np.zeros(n)
    
    Sequence = np.random.choice(n, n, replace=False)

    for i in Sequence:
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
    
    resource_usage = ini_res_usage(X_t_ik, R, T)
    exam_type_patterns, exam_type_deltas = precompute_exam_type_patterns(pro_i, X_t_ik, S_ik, F_ik, R)

    c_in = initial_temp(pro_time, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, punish, ratio, nei_1, nei_2)
    print("c_in", c_in)
    
    start_time_main = time.time()
    results = Parallel(n_jobs=n_threads)(delayed(SA)(pro_time, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, c_in, seed, Lk, freeze_crit, temp_red, punish, ratio, nei_1, nei_2) for seed in range(n_runs))
    best_cost = float('inf')
    
    time_array = []
    cost_array = []
    
    for result in results:
        S_alt, F_alt, Y_alt, c_alt, resource_usage_alt, time_alt = result
        vec_violations = np.count_nonzero(resource_usage_alt > threshold)
        cost_alt = -np.where(Y_alt == 1)[0].size + 10 * vec_violations
        time_array.append(time_alt)
        cost_array.append(cost_alt)
        if cost_alt < best_cost:
            S_ik, F_ik, Y_i, resource_usage = deep_copy(S_alt, F_alt, Y_alt, resource_usage_alt)
            best_cost = cost_alt
    
    print("Runtime: ", time.time() - start_time_main)
    print("lowest_cost", best_cost)
    
    time_array = np.array(time_array)    
    cost_array = np.array(cost_array)
    
    reshaped = cost_array.reshape(-1, n_threads)
    min_per_group = reshaped.min(axis=1)
    reshaped = time_array.reshape(-1, n_threads)
    max_per_group = reshaped.max(axis=1)
    
    print(max_per_group, min_per_group)
    
    print("mean_cost", np.mean(cost_array))
    print("min_costs", np.mean(min_per_group))    

    print("mean_time", np.mean(time_array))
    print("max_time", np.mean(max_per_group))
    
    figplot(S_ik, F_ik, n, T, max_cols)
    plot_resource_utilization(rec_max_p, T, resource_usage, rec_lpk, pro_i)

    return S_ik, F_ik, Y_i, best_cost, time.time() - start_time_main, resource_usage

for Lk in [100, 500, 1000]:
    for freeze_crit in [10, 55, 100]:
        for temp_red in [0.8, 0.9, 0.975]:
            for punish in [0]:
                for ratio in [0.5, 0.7, 0.9]:
                    print(Lk, freeze_crit, temp_red, punish, ratio)
                    _, _, _, _, _, _ = main(100, 10, Lk, freeze_crit, temp_red, punish, ratio, 0.05, 0.05, 487, np.array([3, 5, 10, 1, 10])*2)
                    print("\n")