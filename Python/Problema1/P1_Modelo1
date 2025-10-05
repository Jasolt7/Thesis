# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 14:28:24 2025

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
    count = 0
    for i in range(duration):
        for j in range(n_rows):
            if resource_usage[t + i, j] + delta[i, j] > threshold[0, j]:
                count += 1
    return count

@njit
def earliest_candidate(resource_usage, delta, threshold, start, end, pri1, pri2):
    duration = delta.shape[0]
    n_rows = threshold.shape[1]
    # Full search, but check priority resources first (usually much cheaper)
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
    
    for i in Sequence:
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        duration_glob = sum(rec_lpk[pro_i[i],1])

        t = earliest_candidate(resource_usage, delta, threshold, start, end, pri_s_cam_gama, pri_s_tom)
        starting_time.append(t)
        resource_usage[t : t + duration_glob, :] += delta

    S_ik, F_ik = get_start_end(Sequence, starting_time, pro_i, rec_lpk, n_jobs, max_cols)
    
    makespan = int(max(F_ik[:, -1]))
    
    return S_ik, F_ik, resource_usage, makespan

def initial_solution(threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas):
    order = np.zeros(n)
    for i in range(n):
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        order[i] = delta.shape[0]
        
    order = np.flip(np.argsort(order))
    Sequence = order[0:2]   # start with first two
    S_ik, F_ik, resource_usage, makespan = define_sequence_LS(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
    
    for step in range(2, len(pro_i)):
        Sequence_size = len(Sequence) + 1
        best_makespan = float("inf")
        best_seq = None
        
        for l in range(Sequence_size):
            Sequence_alt = np.insert(Sequence, l, order[step])  # candidate
            _, _, _, makespan_alt = define_sequence_LS(Sequence_alt, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
            if makespan_alt < best_makespan:
                best_makespan = makespan_alt
                best_seq = Sequence_alt.copy()
        
        Sequence = best_seq  # update only once, after testing all insertions
    
    return Sequence

def sa_iteration(S_ik, F_ik, resource_usage, c, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish):
    F_temp, RU_temp = deep_copy(F_ik[:, -1], resource_usage)
    F_best, RU_best = deep_copy(F_ik[:, -1], resource_usage)

    durations = np.zeros(F_temp.shape[0])
    for i in range(F_temp.shape[0]):
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        durations[i] = delta.shape[0]
    
    start, end = compute_makespan(F_temp, T, pro_i, durations)
    makespan = end - start
    vec_violations = np.count_nonzero(RU_temp > threshold)
    current_cost = makespan + punish * vec_violations
    best_cost = current_cost
    
    accepted = 0
    
    i_it = np.random.randint(n, size=(Lk))
    q_it = np.random.rand(Lk)
    shift_it = np.random.rand(Lk)
    start = 0
    end = T
    average_dif = 0
        
    for it in range(Lk):
        i = i_it[it]
        # Look up the precomputed pattern and delta for exam i by its type.
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        # Get the current boundaries for exam i.
        original_finish = int(F_temp[i])
        original_start = original_finish - delta.shape[0]

        # Choose a shift so that the exam remains within [start, end].
        shift = int(start + (end + 1 - delta.shape[0] - start) * shift_it[it] - original_start)
        
        RU_temp = shift_resource_usage_numba(RU_temp, delta, original_start, shift)
        
        # Update start and finish times.
        F_temp[i] += shift
        
        # Insert the exam pattern into the new location.
        new_start = original_start + shift
        new_finish = original_finish + shift
        
        start, end = compute_makespan(F_temp, T, pro_i, durations)
        makespan = end - start

        vec_violations = vec_vio(RU_temp, start, end, threshold)
        alt_cost = makespan + punish * vec_violations
        
        average_dif = average_dif + np.abs((current_cost - alt_cost))
        
        # Acceptance criterion of the SA algorithm.
        if alt_cost < current_cost or q_it[it] < 2.718281828459045**((current_cost - alt_cost) / c):
            current_cost = alt_cost
            if current_cost < best_cost:
                best_cost = current_cost
                RU_best = RU_temp.copy()
                F_best = F_temp.copy()
            accepted += 1

        else:
            # Rollback the shift.
            RU_temp = shift_resource_usage_numba(RU_temp, delta, new_start, -shift)
            F_temp[i] -= shift
    
    F_last_temp = F_temp.copy()
    F_last_best = F_best.copy()
    
    F_temp = np.zeros((n, max_cols))
    S_temp = np.zeros((n, max_cols))
    
    F_best = np.zeros((n, max_cols))
    S_best = np.zeros((n, max_cols))
    
    for i in range(n):
        F_temp[i] = F_last_temp[i] - (F_ik[i, -1] - F_ik[i, :])
        S_temp[i] = F_temp[i] - rec_lpk[pro_i[i], 1]
        
        F_best[i, :] = F_last_best[i] - (F_ik[i, -1] - F_ik[i, :])
        S_best[i] = F_best[i] - rec_lpk[pro_i[i], 1]
        
    return S_temp, F_temp, current_cost, accepted, RU_temp, S_best, F_best, RU_best, best_cost, average_dif

def initial_temp(pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, punish, ratio):
    np.random.seed(seed)

    Sequence = initial_solution(threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas)
    S_ik, F_ik, resource_usage, makespan = define_sequence_LS(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
    
    Lk = 100000
    _, _, _, acc, _, _, _, _, _, average_dif = sa_iteration(S_ik, F_ik, resource_usage, float("inf"), pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish)
    c_in = np.abs((average_dif/Lk)/np.log(ratio))
    return c_in

def SA(pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, c_in, seed, Lk, freeze_crit, temp_red, punish, ratio):
    np.random.seed(seed)
    
    start_time = time.time()

    c = c_in
    best_cost = float('inf')
    
    X_t_ik = np.zeros((T, n, max_cols))
    S_ik = np.zeros((n, max_cols))
    F_ik = np.zeros((n, max_cols))
    
    for i in range(n):
        t = np.random.randint(0, T - sum(rec_lpk[pro_i[i], 1]))
        S_ik[i, 0] = t
        F_ik[i, 0] = S_ik[i, 0] + rec_lpk[pro_i[i], 1, 0]

        for t_prime in range(int(S_ik[i, 0]), int(F_ik[i, 0])):
            X_t_ik[t_prime, i, 0] = 1
        
        for k in range(1, max_cols):
            S_ik[i, k] = F_ik[i, k-1]
            F_ik[i, k] = S_ik[i, k] + rec_lpk[pro_i[i], 1, k]
            for t_prime in range(int(S_ik[i, k]), int(F_ik[i, k])):
                X_t_ik[t_prime, i, k] = 1
        t = F_ik[i, max_cols-1]
    
    resource_usage = ini_res_usage(X_t_ik, R, T)

    freeze = 0

    while freeze < freeze_crit:

        S_new, F_new, cost_new, accepted, RU_new, S_best, F_best, RU_best, best_cost_new, _ = sa_iteration(S_ik, F_ik, resource_usage, c, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish)
        if best_cost_new < best_cost:
            bestS, bestF, bestRU = deep_copy(S_best, F_best, RU_best)
            best_cost = best_cost_new
            freeze = 0
        else:
            freeze += 1
        
        S_ik, F_ik, resource_usage = deep_copy(S_new, F_new, RU_new)
            
        c *= temp_red
        if c < 9.608478221092565e-100:
            c = 9.608478221092565e-100
    
    end_time = time.time() - start_time
    
    return bestS, bestF, c, bestRU, end_time

def main(n_runs, n_threads, Lk, freeze_crit, temp_red, punish, ratio, pro_i):    
    seed = 50
    np.random.seed(seed)  # Global seed
    
    pro_i = np.array(pro_i)
    pro_i = np.repeat(np.arange(len(pro_i)), pro_i)
    rec_lpk = np.load('rec_lpk.npy')
    rec_max_p = np.array([0, 0, 4, 4, 4, 2, 1, 2, 1, 4, 1, 2, 3, 1, 1, 2, 1, 6, 1, 3])
    
    n = pro_i.shape[0]
    n_exams = rec_lpk.shape[0]
    max_cols = rec_lpk.shape[2]
    max_rows = rec_lpk.shape[1]

    T = int(np.sum(rec_lpk[pro_i,1,:]))
    R = rec_lpk[pro_i, 2:, :]
    threshold = rec_max_p[2:].reshape(1, -1)

    X_t_ik = np.zeros((T, n, max_cols))
    S_ik = np.zeros((n, max_cols))
    F_ik = np.zeros((n, max_cols))
    
    for i in range(n):
        t = np.random.randint(0, T - sum(rec_lpk[pro_i[i], 1]))
        S_ik[i, 0] = t
        F_ik[i, 0] = S_ik[i, 0] + rec_lpk[pro_i[i], 1, 0]

        for t_prime in range(int(S_ik[i, 0]), int(F_ik[i, 0])):
            X_t_ik[t_prime, i, 0] = 1
        
        for k in range(1, max_cols):
            S_ik[i, k] = F_ik[i, k-1]
            F_ik[i, k] = S_ik[i, k] + rec_lpk[pro_i[i], 1, k]
            for t_prime in range(int(S_ik[i, k]), int(F_ik[i, k])):
                X_t_ik[t_prime, i, k] = 1
        t = F_ik[i, max_cols-1]
    
    exam_type_patterns, exam_type_deltas = precompute_exam_type_patterns(pro_i, X_t_ik, S_ik, F_ik, R)
    resource_usage = ini_res_usage(X_t_ik, R, T)
    
    c_in = initial_temp(pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, punish, ratio)
    print("c_in", c_in)
    
    start_time_main = time.time()
    results = Parallel(n_jobs=n_threads)(delayed(SA)(pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, c_in, seed, Lk, freeze_crit, temp_red, punish, ratio) for seed in range(n_runs))
    best_cost = float('inf')
    
    time_array = []
    cost_array = []
    
    for result in results:
        S_alt, F_alt, c_alt, resource_usage_alt, time_alt = result
        
        F_temp = F_alt[:, -1]
        durations = np.zeros(F_temp.shape[0])
        for i in range(F_temp.shape[0]):
            exam_type = pro_i[i]
            delta = exam_type_deltas[exam_type]
            durations[i] = delta.shape[0]
        
        start, end = compute_makespan(F_temp, T, pro_i, durations)
        makespan = end - start

        vec_violations = np.count_nonzero(resource_usage_alt > threshold)
        cost_alt = makespan + 100 * vec_violations
        time_array.append(time_alt)
        cost_array.append(cost_alt)
        if cost_alt < best_cost:
            S_ik, F_ik, resource_usage = deep_copy(S_alt, F_alt, resource_usage_alt)
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
        
    start, end = compute_makespan(F_ik[:, -1], T, pro_i, durations)
    
    S_ik -= start
    F_ik -= start
    resource_usage = resource_usage[int(start): int(end)]
    T = resource_usage.shape[0]
    print(start, end, T)
    figplot(S_ik, F_ik, n, end-start, max_cols)
    plot_resource_utilization(rec_max_p, int(end-start), resource_usage, rec_lpk, pro_i)
    
    return S_ik, F_ik, best_cost, time.time() - start_time_main, resource_usage

#S_ik, F_ik, best_cost, runtime, resource_usage = main(100, 10, 50, 2500, 100, 0.99, 10, 0.9)


for Lk in [500, 1500, 2500]:
    for freeze_crit in [10, 55, 100]:
        for temp_red in [0.8, 0.9, 0.975]:
            for punish in [10, 55, 100]:
                for ratio in [0.5, 0.7, 0.9]:
                    print(Lk, freeze_crit, temp_red, punish, ratio)
                    _, _, _, _, _ = main(100, 10, Lk, freeze_crit, temp_red, punish, ratio, [3, 5, 10, 1, 10])
                    print("\n")
