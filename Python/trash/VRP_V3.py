# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:44:33 2025

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
from functools import lru_cache
from distance_utils import get_distance

def plot_warehouse_path(Sequence, max_rows, max_cols, Lk, freeze_crit, temp_red, ratio, cost, show_crow):
    plt.figure(figsize=(7, 7))
    
    Sequence = [(0,0)] + Sequence + [(0,0)]

    # expand limits slightly to see all lines clearly
    plt.xlim(-0.5, max_cols + 0.5)
    plt.ylim(-0.5, max_rows + 0.5)
    plt.gca().invert_yaxis()  # top-down layout like a floor plan

    # grid lines
    for i in range(max_rows + 2):
        plt.axhline(i - 0.5, color='lightgray', linewidth=0.6)
    for j in range(max_cols + 2):
        plt.axvline(j - 0.5, color='lightgray', linewidth=0.6)

    # main walking path (columns-only)
    for i in range(len(Sequence) - 1):
        x1, y1 = Sequence[i]
        x2, y2 = Sequence[i + 1]

        # columns-only movement
        if x1 != x2:
            y_mid = 0 if (y1 + y2) < max_rows else max_rows
            plt.plot([x1, x1], [y1, y_mid], 'b--', alpha=0.7)
            plt.plot([x1, x2], [y_mid, y_mid], 'b--', alpha=0.7)
            plt.plot([x2, x2], [y_mid, y2], 'b--', alpha=0.7)
        else:
            plt.plot([x1, x2], [y1, y2], 'b--', alpha=0.7)

        # as-the-crow-flies (optional)
        if show_crow:
            plt.plot([x1, x2], [y1, y2], 'r-', alpha=0.4, linewidth=1.2)

        # pick points
        plt.scatter(x1, y1, color='red', s=50, zorder=5)
        plt.text(x1 + 0.1, y1 - 0.2, f'{i}', fontsize=9)

    # last point
    x_last, y_last = Sequence[-1]
    plt.scatter(x_last, y_last, color='green', s=60, zorder=5)
    plt.text(x_last + 0.1, y_last - 0.2, f'{len(Sequence)-1}', fontsize=9)

    plt.title(f"Warehouse Path\nBlue: Walking Path | Red: As-the-Crow-Flies\nLk={Lk}, freeze_crit={freeze_crit}, temp_red={temp_red}, ratio={ratio}, cost={cost}")
    plt.xlabel("Columns (x)")
    plt.ylabel("Rows (y)")
    plt.grid(False)
    plt.show()

def deep_copy(*arrays):
    # Returns copies of all passed arrays.
    return [a.copy() for a in arrays]

def cost_func(Sequence, max_rows):
    dist = 0
    max_storage = 25
    current_storage = 0
    p1 = (0, 0)
    for seq in Sequence:
        if current_storage + seq[2] > max_storage:
            dist += get_distance(p1, (0, 0), max_rows)
            p1 = (0, 0)
            current_storage = 0
        p2 = seq[:2]
        dist += get_distance(p1, p2, max_rows)
        p1 = p2
        current_storage += seq[2]
        
    dist += get_distance(p1, (0, 0), max_rows)
    return dist

def sa_iteration(Sequence, max_rows, c, Lk):
    Sequence_best = Sequence.copy()
    current_cost = cost_func(Sequence, max_rows)
    best_cost = current_cost
        
    i_it = (np.random.rand(Lk, 2)*len(Sequence)).astype(int)
    q_it = np.random.rand(Lk)
    
    average_dif = 0
    neighbour = np.random.choice(1, size=Lk)
    
    
    for it in range(Lk):
        i_st = i_it[it, 0]
        i_nd = i_it[it, 1]
        while i_st == i_nd:
            i_nd = np.random.choice(len(Sequence))  
            
        if neighbour[it] == 0:   
            Sequence[i_st], Sequence[i_nd] = Sequence[i_nd], Sequence[i_st]
            
        elif neighbour[it] == 1:
            holder = Sequence.pop(i_st)
            Sequence.insert(i_nd, holder)
        
        elif neighbour[it] == 2:
            i_st, i_nd = sorted([i_st, i_nd])
            Sequence[i_st:i_nd+1] = reversed(Sequence[i_st:i_nd+1])
        
        alt_cost = cost_func(Sequence, max_rows)
        average_dif = average_dif + np.abs((current_cost - alt_cost))

        # Acceptance criterion of the SA algorithm.
        if alt_cost < current_cost or q_it[it] < 2.718281828459045**((current_cost - alt_cost) / c):
            current_cost = alt_cost
            if current_cost < best_cost:
                best_cost = current_cost
                Sequence_best = Sequence.copy()

        else:
            if neighbour[it] == 0:
                Sequence[i_st], Sequence[i_nd] = Sequence[i_nd], Sequence[i_st]
                
            elif neighbour[it] == 1:
                holder = Sequence.pop(i_nd)
                Sequence.insert(i_st, holder)
                
            elif neighbour[it] == 2:
                Sequence[i_st:i_nd+1] = reversed(Sequence[i_st:i_nd+1])
    
    return Sequence, current_cost, Sequence_best, best_cost, average_dif

def initial_temp(max_rows, max_cols, seed, Lk, ratio):
    np.random.seed(seed)
    coords = np.array([(r, c) for r in range(max_rows) for c in range(max_cols)])
    idx = np.random.choice(len(coords), size=100, replace=False)
    Sequence = [(49, 12, 1),(22, 7, 4),(25, 60, 2),(39, 2, 2),(37, 15, 1),(99, 95, 5),(40, 80, 1),(77, 24, 3),(76, 32, 5),(31, 43, 4),(34, 13, 5),(35, 59, 1),(14, 97, 4),(97, 8, 4),(90, 29, 4),(55, 23, 2),(31, 51, 3),(88, 4, 1),(67, 26, 5),(39, 67, 2),(50, 78, 3),(52, 36, 4),(92, 11, 4),(57, 16, 2),(66, 1, 4)]
    
    _ ,_ , _, _, average_dif = sa_iteration(Sequence, max_rows, float("inf"), Lk)

    c_in = np.abs((average_dif/Lk)/np.log(ratio))
    return c_in

def SA(max_rows, max_cols, c_in, seed, Lk, freeze_crit, temp_red, ratio):
    np.random.seed(seed)
    start_time = time.time()
    
    coords = np.array([(r, c) for r in range(max_rows) for c in range(max_cols)])
    idx = np.random.choice(len(coords), size=100, replace=False)
    Sequence = [tuple(coords[i]) for i in idx]
    Sequence = [(49, 12, 1),(22, 7, 4),(25, 60, 2),(39, 2, 2),(37, 15, 1),(99, 95, 5),(40, 80, 1),(77, 24, 3),(76, 32, 5),(31, 43, 4),(34, 13, 5),(35, 59, 1),(14, 97, 4),(97, 8, 4),(90, 29, 4),(55, 23, 2),(31, 51, 3),(88, 4, 1),(67, 26, 5),(39, 67, 2),(50, 78, 3),(52, 36, 4),(92, 11, 4),(57, 16, 2),(66, 1, 4)]
    
    c = c_in
    best_cost = float('inf')
    
    freeze = 0

    while freeze < freeze_crit:

        Sequence ,_ , newSequence, new_cost, _ = sa_iteration(Sequence, max_rows, c, int(Lk/10))
        if new_cost < best_cost:
            bestSequence = deep_copy(newSequence)
            best_cost = new_cost
            freeze = 0
        else:
            freeze += 1
            
        c *= temp_red
        if c < 9.608478221092565e-100:
            c = 9.608478221092565e-100
        
    return bestSequence[0], best_cost, time.time() - start_time

def main(n_runs, n_threads, max_rows, max_cols, seed, Lk, freeze_crit, temp_red, ratio):
    np.random.seed(seed)
    
    c_in = initial_temp(max_rows, max_cols, seed, Lk, ratio)
    print(c_in)
    
    results = Parallel(n_jobs=n_threads)(delayed(SA)(max_rows, max_cols, c_in, seed, Lk, freeze_crit, temp_red, ratio) for seed in range(n_runs))
    
    cost_avg = 0
    time_avg = 0
    best_cost_ever = float("inf")
    
    for result in results:
        bestSequence, best_cost, time = result
        cost_avg += best_cost
        time_avg += time
        if best_cost < best_cost_ever:
            best_cost_ever = best_cost
            bestSequenceever = bestSequence
        print("[",best_cost, ",", time, "]")
    print("\n", Lk, freeze_crit, temp_red, ratio)    
    print("[",cost_avg/n_runs, ",", time_avg/n_runs, "]")

    #plot_warehouse_path(bestSequenceever, max_rows, max_cols, Lk, freeze_crit, temp_red, ratio, best_cost, True)

if __name__ == "__main__":
    main(10, 10, 100, 100, 50, int(100 * (25 * (25 - 1) / 2)), 50, 0.8, 0.5)