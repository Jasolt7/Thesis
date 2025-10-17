# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:45:17 2025

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
import itertools
import math

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

def distance_func(p1, p2, max_rows):
    x1, y1 = p1
    x2, y2 = p2

    h_dist = abs(x1 - x2)
    if h_dist != 0:
        v_dist = min(2*max_rows - y1 - y2, (y1 + y2))
    else:
        v_dist = abs(y1 - y2)

    return h_dist + v_dist

@njit
def membership(val, array):
    for i in array:
        if val == i:
            return True
    return False

@lru_cache(maxsize=None)
def get_distance(p1, p2, max_rows):
    return distance_func(p1, p2, max_rows)

def cost_func(Sequence, max_rows):
    dist = 0
    p1 = (0, 0)
    for seq in Sequence:
        p2 = seq
        dist += get_distance(p1, p2, max_rows)
        p1 = p2
    dist += get_distance(p1, (0, 0), max_rows)
    return dist

def crossover_OX(Parent1, Parent2, n):
    Off_spring1 = np.zeros(n).astype(int)-1
    in_array1 = []
    Off_spring2 = np.zeros(n).astype(int)-1
    in_array2 = []
    points = np.sort(np.random.choice(n, size=2, replace=False))
    #[Sequence[i] for i in randomSeq]
    for i in range(points[0], points[1]):
        Off_spring1[i] = Parent1[i]
        in_array1.append(Parent1[i]) 
        Off_spring2[i] = Parent2[i]
        in_array2.append(Parent2[i])
    
    idx_1 = points[1]-n
    idx_2 = points[1]-n
    for j in range(points[1]-n, points[1]):
        if not membership(Parent2[j], in_array1):
            Off_spring1[idx_1] = Parent2[j]
            idx_1 += 1
            
        if not membership(Parent1[j], in_array2):
            Off_spring2[idx_2] = Parent1[j]
            idx_2 += 1
    
    return Off_spring1, Off_spring2

def crossover_SCX(Sequence, Parent1, Parent2, n, max_rows):
    offspring1 = [Parent1[0]]
    visited1 = {Parent1[0]}
    offspring2 = [Parent2[0]]
    visited2 = {Parent2[0]}
    
    pos1 = {Parent1[i]: i for i in range(n)}
    pos2 = {Parent2[i]: i for i in range(n)}

    for i in range(1, n):
        current = offspring1[-1]
        p0 = Sequence[current]
        
        right_city = Parent1[(pos1[current] + 1) % n]
        if right_city not in visited1:
            p1 = Sequence[right_city]
            idx1 = right_city
        else:
            idx1 = next(j for j in Parent1 if j not in visited1)
        
        right_city = Parent2[(pos2[current] + 1) % n]
        if right_city not in visited1:
            p2 = Sequence[right_city]
            idx2 = right_city
        else:
            idx2 = next(j for j in Parent2 if j not in visited1)
                            
        if get_distance(p0, p1, max_rows) < get_distance(p0, p2, max_rows):
            offspring1.append(idx1)
            visited1.add(idx1)
        else:
            offspring1.append(idx2)
            visited1.add(idx2)

    
    for i in range(1, n):
        current = offspring2[-1]
        p0 = Sequence[current]
        
        right_city = Parent1[(pos1[current] + 1) % n]
        if right_city not in visited2:
            p1 = Sequence[right_city]
            idx1 = right_city
        else:
            idx1 = next(j for j in Parent1 if j not in visited2)
        
        right_city = Parent2[(pos2[current] + 1) % n]

        if right_city not in visited2:
            p2 = Sequence[right_city]
            idx2 = right_city
        else:
            idx2 = next(j for j in Parent2 if j not in visited2)
                            
        if get_distance(p0, p1, max_rows) < get_distance(p0, p2, max_rows):
            offspring2.append(idx1)
            visited2.add(idx1)
        else:
            offspring2.append(idx2)
            visited2.add(idx2)
            
    return offspring1, offspring2

def ge_pop(pop_size, n):
    return [np.random.choice(n, n, replace=False).astype(int) for _ in range(pop_size)]

def evaluate_pop(population, Sequence, max_rows):
    cost_array = np.zeros(len(population))
    
    for i in range(len(population)):
        eval_seq = [Sequence[j] for j in population[i]]
        cost_array[i] = cost_func(eval_seq, max_rows)
    
    return cost_array

def mutation(off_spring, n):
    i = np.sort(np.random.choice(n, size=2, replace=False))
    i_st = i[0]
    i_nd = i[1]
    holder = off_spring[i_st]
    off_spring[i_st] = off_spring[i_nd]
    off_spring[i_nd] = holder
    
    return off_spring

def thread_assign(Sequence, seed, pop_size, freeze_crit, n, max_rows, mut_prob):
    np.random.seed(seed)
    parents = ge_pop(pop_size, n)
    off_springs = []
    freeze = 0
    best_cost = float("inf")
    start_time = time.time()

    while freeze < freeze_crit:
        cost_array = evaluate_pop(parents, Sequence, max_rows)
        idx = np.argsort(cost_array)
        if cost_array[idx[0]] < best_cost:
            best_solution = parents[idx[0]]
            best_cost = cost_array[idx[0]]
            freeze = 0
        else:
            freeze += 1
        for l in range(0, pop_size, 2):
            Off_spring1, Off_spring2 = crossover_SCX(Sequence, parents[idx[l]], parents[idx[l+1]], n, max_rows)
            #Off_spring1, Off_spring2 = crossover_OX(parents[idx[l]], parents[idx[l+1]], n)
            mut = np.random.choice(2, p=[1-mut_prob, mut_prob])
            if mut == 1:
                off_springs.append(mutation(Off_spring1, n))
                off_springs.append(mutation(Off_spring2, n))
            else:
                off_springs.append(Off_spring1)
                off_springs.append(Off_spring2)
        parents = off_springs
    
    if cost_array[idx[0]] < best_cost:
        best_solution = parents[idx[0]]
        best_cost = cost_array[idx[0]]
    
    return best_cost, best_solution, time.time() - start_time
        
def main(Sequence, n_runs, n_threads, pop_size, freeze_crit, n, max_rows, mut_prob):
    
    results = Parallel(n_jobs=n_threads,  backend="threading")(delayed(thread_assign)(Sequence, seed, pop_size, freeze_crit, n, max_rows, mut_prob) for seed in range(n_runs))
    
    cost_avg = 0
    time_avg = 0
    
    for result in results:
        best_cost, best_solution, time = result
        cost_avg += best_cost
        time_avg += time
        #print("[",best_cost, ",", time, "]")
    print(pop_size, freeze_crit, mut_prob)    
    print("[",cost_avg/n_runs, ",", time_avg/n_runs, "]")

print(main([(49, 12),(22, 7),(25, 60),(39, 2),(37, 15),(99, 95),(40, 80),(77, 24),(76, 32),(31, 43),(34, 13),(35, 59),(14, 97),(97, 8),(90, 29),(55, 23),(31, 51),(88, 4),(67, 26),(39, 67),(50, 78),(52, 36),(92, 11),(57, 16),(66, 1)], 
     50, 10, 500, 5, 25, 100, 0.0))






