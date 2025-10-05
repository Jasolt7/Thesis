"""
Created on Tue Aug 19 15:07:23 2025

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

        t = earliest_candidate(resource_usage, delta, threshold, start, end, pri_s_cam_gama, pri_s_tom, exam_type, exam_type_start)
        if t > exam_type_start[exam_type]:
            exam_type_start[exam_type] = t
        starting_time.append(t)
        resource_usage[t : t + duration_glob, :] += delta

    S_ik, F_ik = get_start_end(Sequence, starting_time, pro_i, rec_lpk, n_jobs, max_cols)

    makespan = int(max(F_ik[:, -1]))

    return S_ik, F_ik, resource_usage, makespan

def define_sequence_ND(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, start, end):
    Sequence = np.asarray(Sequence).ravel().astype(int)  # sanitize
    resource_usage = np.zeros((T, rec_lpk.shape[1]-2))
    starting_time = []
    n_jobs = n
    n = len(Sequence)
    pri_s_cam_gama = 8
    pri_s_tom = 16    
    exam_type_start = [0]*np.unique(pro_i)
    
    start = 0
    for i in Sequence:
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        duration_glob = sum(rec_lpk[pro_i[i],1])

        t = earliest_candidate(resource_usage, delta, threshold, start, end, pri_s_cam_gama, pri_s_tom, exam_type, exam_type_start)
        if t > exam_type_start[exam_type]:
            exam_type_start[exam_type] = t
        start = t
        starting_time.append(t)
        resource_usage[t : t + duration_glob, :] += delta

    S_ik, F_ik = get_start_end(Sequence, starting_time, pro_i, rec_lpk, n_jobs, max_cols)
    
    makespan = int(max(F_ik[:, -1]))
    
    return S_ik, F_ik, resource_usage, makespan

def define_sequence_ELS(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, start, end):
    Sequence = np.asarray(Sequence).ravel().astype(int)  # sanitize
    resource_usage_1 = np.zeros((T, rec_lpk.shape[1]-2))
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

        t = earliest_candidate(resource_usage_1, delta, threshold, start, end, pri_s_cam_gama, pri_s_tom, exam_type, exam_type_start)
        if t > exam_type_start[exam_type]:
            exam_type_start[exam_type] = t       
        starting_time.append(t)
        resource_usage_1[t : t + duration_glob, :] += delta

    S_ik_1, F_ik_1 = get_start_end(Sequence, starting_time, pro_i, rec_lpk, n_jobs, max_cols)
    makespan_1 = int(max(F_ik_1[:, -1]))
    
    resource_usage_2 = np.zeros((T, rec_lpk.shape[1]-2))
    Sequence_alt = np.flip(Sequence)
    starting_time = []
    exam_type_start = [0]*np.unique(pro_i)
    for i in Sequence_alt:
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        delta = np.flipud(delta)
        duration_glob = sum(rec_lpk[pro_i[i],1])

        t = earliest_candidate(resource_usage_2, delta, threshold, start, end, pri_s_cam_gama, pri_s_tom, exam_type, exam_type_start)
        if t > exam_type_start[exam_type]:
            exam_type_start[exam_type] = t        
        starting_time.append(t)
        resource_usage_2[t : t + duration_glob, :] += delta
    
    S_ik_2, F_ik_2 = get_start_end(Sequence_alt, starting_time, pro_i, rec_lpk, n_jobs, max_cols)
    makespan_2 = int(max(F_ik_2[:, -1]))
    
    if makespan_1 < makespan_2:
        S_ik = S_ik_1.copy()
        F_ik = F_ik_1.copy()
        makespan = makespan_1
        resource_usage = resource_usage_1.copy()
    else:
        S_ik = S_ik_2.copy()
        F_ik = F_ik_2.copy()
        makespan = makespan_2
        resource_usage = resource_usage_2.copy()
    
    return S_ik, F_ik, resource_usage, makespan

def define_sequence_END(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, start, end):
    Sequence = np.asarray(Sequence).ravel().astype(int)  # sanitize
    resource_usage_1 = np.zeros((T, rec_lpk.shape[1]-2))
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

        t = earliest_candidate(resource_usage_1, delta, threshold, start, end, pri_s_cam_gama, pri_s_tom, exam_type, exam_type_start)
        start = t
        if t > exam_type_start[exam_type]:
            exam_type_start[exam_type] = t       
        starting_time.append(t)
        resource_usage_1[t : t + duration_glob, :] += delta

    S_ik_1, F_ik_1 = get_start_end(Sequence, starting_time, pro_i, rec_lpk, n_jobs, max_cols)
    makespan_1 = int(max(F_ik_1[:, -1]))
    
    resource_usage_2 = np.zeros((T, rec_lpk.shape[1]-2))
    Sequence_alt = np.flip(Sequence)
    starting_time = []
    exam_type_start = [0]*np.unique(pro_i)
    for i in Sequence_alt:
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        delta = np.flipud(delta)
        duration_glob = sum(rec_lpk[pro_i[i],1])

        t = earliest_candidate(resource_usage_2, delta, threshold, start, end, pri_s_cam_gama, pri_s_tom, exam_type, exam_type_start)
        start = t
        if t > exam_type_start[exam_type]:
            exam_type_start[exam_type] = t        
        starting_time.append(t)
        resource_usage_2[t : t + duration_glob, :] += delta
    
    S_ik_2, F_ik_2 = get_start_end(Sequence_alt, starting_time, pro_i, rec_lpk, n_jobs, max_cols)
    makespan_2 = int(max(F_ik_2[:, -1]))
    
    if makespan_1 < makespan_2:
        S_ik = S_ik_1.copy()
        F_ik = F_ik_1.copy()
        makespan = makespan_1
        resource_usage = resource_usage_1.copy()
    else:
        S_ik = S_ik_2.copy()
        F_ik = F_ik_2.copy()
        makespan = makespan_2
        resource_usage = resource_usage_2.copy()
    
    return S_ik, F_ik, resource_usage, makespan

def initial_solution(threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas):
    order = np.zeros(n)
    for i in range(n):
        exam_type = pro_i[i]
        delta = exam_type_deltas[exam_type]
        order[i] = delta.shape[0]
        
    order = np.flip(np.argsort(order))
    Sequence = order[0:2]   # start with first two
    _, _, _, makespan = define_sequence_LS(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
    
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

def sa_iteration(Sequence, c, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish):
    Sequence_best = deep_copy(Sequence)
    S_ik, F_ik, resource_usage, makespan = define_sequence_LS(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
    current_cost = makespan
    best_cost = current_cost
    
    accepted = 0
    
    i_it = np.array([np.random.choice(n, size=2, replace=False) for _ in range(Lk)])
    q_it = np.random.rand(Lk)
    
    average_dif = 0
    
    for it in range(Lk):
        i_st = i_it[it, 0]
        i_nd = i_it[it, 1]
        while pro_i[Sequence[i_st]] == pro_i[Sequence[i_nd]]:
            new_i = np.random.choice(n, size=2, replace=False)
            i_st = new_i[0]
            i_nd = new_i[1]
            
        holder = Sequence[i_st]
        Sequence[i_st] = Sequence[i_nd]
        Sequence[i_nd] = holder
        #print(Sequence)
        S_ik, F_ik, resource_usage, makespan = define_sequence_LS(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
        #print(makespan)
        alt_cost = makespan

        average_dif = average_dif + np.abs((current_cost - alt_cost))

        # Acceptance criterion of the SA algorithm.
        if alt_cost < current_cost or q_it[it] < 2.718281828459045**((current_cost - alt_cost) / c):
            current_cost = alt_cost
            if current_cost < best_cost:
                best_cost = current_cost
                Sequence_best = Sequence.copy()
            accepted += 1

        else:
            holder = Sequence[i_st]
            Sequence[i_st] = Sequence[i_nd]
            Sequence[i_nd] = holder
    
    return Sequence, current_cost, accepted, Sequence_best, best_cost, S_ik, F_ik, resource_usage, average_dif

def initial_temp(pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, punish, ratio):
    np.random.seed(seed)

    Sequence = initial_solution(threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas)
    
    Lk = 1000
    _, _, acc, _, _, _, _, _, average_dif = sa_iteration(Sequence, float("inf"), pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish)
    c_in = np.abs((average_dif/Lk)/np.log(ratio))
    return c_in

def SA(pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, c_in, seed, Lk, freeze_crit, temp_red, punish, ratio):
    np.random.seed(seed)
    start_time = time.time()

    c = c_in
    best_cost = float('inf')
    
    Sequence = np.random.choice(n, n, replace=False)

    freeze = 0

    while freeze < freeze_crit:

        Sequence, _, _, newSequence, _, _, _, _, _ = sa_iteration(Sequence, c, pro_i, rec_lpk, rec_max_p, n, n_exams, max_cols, max_rows, T, R, threshold, exam_type_deltas, seed, Lk, punish)
        newS, newF, newRU, makespan = define_sequence_LS(newSequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
        new_cost = makespan
        if new_cost < best_cost:
            bestSequence, bestS, bestF, bestRU = deep_copy(newSequence, newS, newF, newRU)
            best_cost = new_cost

            freeze = 0
        else:
            freeze += 1
            
        c *= temp_red
        if c < 9.608478221092565e-100:
            c = 9.608478221092565e-100
    
    end_time = time.time() - start_time

    return bestSequence, bestS, bestF, bestRU, end_time

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
    
    t = 0
    it = 0
    for i in range(n):
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
        it += 1
    
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
        Sequcne_alt, _, _, _, time_alt = result
        S_alt, F_alt, RU_alt, makespan = define_sequence_LS(Sequcne_alt, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
        cost_alt = int(max(F_alt[:, -1]))
        time_array.append(time_alt)
        cost_array.append(cost_alt)
        if cost_alt < best_cost:
            S_ik, F_ik, Sequence, resource_usage = deep_copy(S_alt, F_alt, Sequcne_alt, RU_alt)
            best_cost = cost_alt
    
    S_ik, F_ik, resource_usage, makespan = define_sequence_LS(Sequence, threshold, T, pro_i, rec_lpk, n, max_cols, exam_type_deltas, 0, T)
    
    makespan = int(max(F_ik[:, -1]))
    cost = makespan
    
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
    
    figplot(S_ik, F_ik, n, makespan, max_cols)
    plot_resource_utilization(rec_max_p, makespan, resource_usage, rec_lpk, pro_i)
    
    return Sequence, S_ik, F_ik, cost, time.time() - start_time_main, resource_usage


_, _, _, _, _, _ = main(100, 10, 250, 55, 0.975, 0, 0.5, [3, 5, 10, 1, 10])

"""
for Lk in [50, 150, 250]:
    for freeze_crit in [5, 30, 55]:
        for temp_red in [0.8, 0.9, 0.975]:
            for punish in [0]:
                for ratio in [0.5, 0.7, 0.9]:
                    print(Lk, freeze_crit, temp_red, punish, ratio)
                    _, _, _, _, _, _ = main(100, 10, Lk, freeze_crit, temp_red, punish, ratio, [3, 5, 10, 1, 10])
                    print("\n")
"""
