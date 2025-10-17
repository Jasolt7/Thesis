#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <cstdint>  // For `uint8_t`, `int8_t`, etc.
#include <cstdio>   // For `printf()`
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>
#include <future>
#include <vector>
#include <unordered_set>
#include <sstream>

using Sequence_vec = std::vector<std::vector<int>>;
using coord = std::vector<int>;


int distance_func(int x1, int y1, int x2, int y2, int max_rows) {
    int v_dist = 10000000;
    int h_dist = std::abs(x1 - x2);
    int hallway[] = {0, max_rows};

    if (h_dist != 0) {
        for (int h : hallway) {
            int dist_via_hall = std::abs(h - y1) + std::abs(h - y2);
            if (dist_via_hall < v_dist) {
                v_dist = dist_via_hall;
            }
        }
    } else {
        v_dist = std::abs(y1 - y2);
    }

    return h_dist + v_dist;
}

int cost_func(const Sequence_vec& Sequence, int n, int max_rows) {
    int dist = 0;
    int max_storage = 10;
    int current_storage = 0;

    int x1 = 0, x2;
    int y1 = 0, y2;
    for (int i = 0; i < n; i++) {
        if (current_storage == max_storage) {
            dist += distance_func(x1, y1, 0, 0, max_rows);
            x1 = 0;
            y1 = 0;
            current_storage = 0;
        }
        x2 = Sequence[i][0];
        y2 = Sequence[i][1];
        dist += distance_func(x1, y1, x2, y2, max_rows);
        x1 = x2;
        y1 = y2;
        current_storage++;
    }
    x2 = 0;
    y2 = 0;
    dist += distance_func(x1, y1, x2, y2, max_rows);

    return dist;
}

struct SAResults {
    Sequence_vec Sequence_best;
    Sequence_vec Sequence;
    float average_dif;
};

SAResults sa_iteration(const Sequence_vec& initial_sequence, int Lk, float c, std::mt19937& g, int n, int max_rows) {
    SAResults results;
    results.Sequence = initial_sequence;          // current solution
    results.Sequence_best = initial_sequence;     // best found so far
    results.average_dif = 0.0f;

    int best_cost = cost_func(results.Sequence, n, max_rows);
    int current_cost = best_cost;

    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    for (int it = 0; it < Lk; it++) {
        int i_st = g() % n;
        int i_nd = g() % n;
        // ensure different positions and different exam types (as original intent)
        while (i_st == i_nd) {
            i_st = g() % n;
            i_nd = g() % n;
        }

        std::swap(results.Sequence[i_st], results.Sequence[i_nd]);

        int alt_cost = cost_func(results.Sequence, n, max_rows);

        results.average_dif += std::abs(alt_cost - current_cost);

        bool accept = false;
        if (alt_cost < current_cost) {
            accept = true;
        } else {
            double prob = std::exp((current_cost - alt_cost) / c);
            double r = unif01(g);
            if (r < prob) accept = true;
        }

        if (accept) {
            current_cost = alt_cost;
            if (current_cost < best_cost) {
                best_cost = current_cost;
                results.Sequence_best = results.Sequence;
            }
        } else {
            // revert swap
            std::swap(results.Sequence[i_st], results.Sequence[i_nd]);
        }
    }

    return results;
}

struct SARuns {
    Sequence_vec Sequence_best;
    int best_cost;
    double time;
    int seed;
};

SARuns SA(int n, int seed, int max_rows, int max_cols, float c_in, int Lk, float cooling_rate, int freeze_crit) {
    SARuns run;
    run.seed = seed;

    // Initialize RNG
    std::mt19937 g(seed);   // Mersenne Twister RNG
    std::uniform_int_distribution<> dist_x(0, max_cols - 1);
    std::uniform_int_distribution<> dist_y(0, max_rows - 1);

    // To avoid duplicates, use a set
    std::unordered_set<std::string> used;
    Sequence_vec Sequence;

    while (Sequence.size() < n) {
        int x = dist_x(g);
        int y = dist_y(g);

        std::ostringstream key;
        key << x << "," << y;

        if (used.insert(key.str()).second) {  // only add if new
            Sequence.push_back({x, y});
        }
    }

    // Print the generated sequence
    for (size_t i = 0; i < Sequence.size(); ++i) {
        printf("(%d, %d),\n", Sequence[i][0], Sequence[i][1]);
    }

    int freeze = freeze_crit;
    Sequence_vec alt_time_best_Sequence(n);
        
    int best_cost = cost_func(Sequence, n, max_rows);
    auto t1 = std::chrono::high_resolution_clock::now();
    while (freeze > 0) {
        SAResults result = sa_iteration(Sequence, Lk, c_in, g, n, max_rows);
        int alt_cost = cost_func(Sequence, n, max_rows);

        Sequence = result.Sequence;

        if (alt_cost < best_cost) {
            freeze = freeze_crit;
            best_cost = alt_cost;
            alt_time_best_Sequence = result.Sequence_best;

        } else {
            freeze--;
        }
        c_in = c_in * cooling_rate;
        printf("%f,%d\n",c_in, best_cost);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;

    run.time = elapsed.count();
    run.best_cost = best_cost;
    run.Sequence_best = alt_time_best_Sequence;

    return run;
}

int main() {
    int seed = 50;
    int max_cols = 500;
    int max_rows = 500;
    int n = 100;

    int Lk = 1 * (max_cols * max_rows);
    int freeze_crit = 50;
    float cooling_rate = 0.8f;
    float ratio = 0.5f;
    float c_in = 688.307251f;
    printf("Initial Temp: %f\n", c_in);

    SARuns run = SA(n, seed, max_rows, max_cols, c_in, Lk, cooling_rate, freeze_crit);

    for (size_t i = 0; i < run.Sequence_best.size(); ++i) {
        printf("(%d, %d),\n", run.Sequence_best[i][0], run.Sequence_best[i][1]);
    }
        printf("\nBest solution found has cost %d, it took %.3fs\n", run.best_cost, run.time);

}