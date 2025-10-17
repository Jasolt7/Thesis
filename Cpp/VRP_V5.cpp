// valgrind --tool=callgrind --simulate-cache=yes ./P1_Modelo3_V5
// callgrind_annotate callgrind.out.<pid> --threshold=20
// kcachegrind callgrind.out.<pid>

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

struct Item {
    int x, y, w;
};

using Sequence_vec = std::vector<Item>;
using coord = std::vector<int>;

int distance_func(int x1, int y1, int x2, int y2, int max_rows) {
    int v_dist = 10000000;
    int h_dist = std::abs(x1 - x2);
    int hallway[] = {0, max_rows};

    if (h_dist != 0) {
        // for (int h : hallway) {
        //     int dist_via_hall = std::abs(h - y1) + std::abs(h - y2);
        //     if (dist_via_hall < v_dist) {
        //         v_dist = dist_via_hall;
        //     }
        // }
        if ((2*max_rows) - y1 - y2 > (y1 + y2)) {
            v_dist = y1 + y2;
        } else {
            v_dist = (2*max_rows) - y1 - y2;
        }
    } else {
        v_dist = std::abs(y1 - y2);
    }

    return h_dist + v_dist;
}

int cost_func(const Sequence_vec& Sequence, int n, int max_rows) {
    int dist = 0;
    int max_storage = 25;
    int current_storage = 0;
    int c_x = 0, y_x = 0;

    int x1 = c_x, x2;
    int y1 = y_x, y2;
    int over_wei;

    for (int i = 0; i < n; i++) {
        x2 = Sequence[i].x;
        y2 = Sequence[i].y;
        if (x2 == 0 && y2 == 0) {
            current_storage = 0;
        } else {
            current_storage += Sequence[i].w;
        }

        if (current_storage > max_storage) {
            over_wei = current_storage - max_storage;
        } else {
            over_wei = 0;
        }

        dist += distance_func(x1, y1, x2, y2, max_rows) + 10*over_wei;
        x1 = x2;
        y1 = y2;
    }
    dist += distance_func(x1, y1, c_x, y_x, max_rows);

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
    std::uniform_int_distribution<int> rand_idx(0, n - 1);

    for (int it = 0; it < Lk; it++) {
        int i_st = rand_idx(g);
        int i_nd = rand_idx(g);
        // ensure different positions and different exam types (as original intent)
        while (i_st == i_nd) {
            i_st = rand_idx(g);
            i_nd = rand_idx(g);
        }

        double nei = unif01(g);

        if (nei < 1.0/3.0) {
            // swap
            std::swap(results.Sequence[i_st], results.Sequence[i_nd]);

        } else if (nei < 2.0/3.0) {
            // insertion
            Item holder = results.Sequence[i_st];
            if (i_st < i_nd) {
                for (int i = i_st; i < i_nd; ++i) {
                    results.Sequence[i] = results.Sequence[i + 1];
                }
            } else {
                for (int i = i_st; i > i_nd; --i) {
                   results.Sequence[i] = results.Sequence[i - 1];
                }
            }
            results.Sequence[i_nd] = holder;

        } else {
            // inversion
            int lo = i_st, hi = i_nd;
            while (lo < hi)
                std::swap(results.Sequence[lo++], results.Sequence[hi--]);
        }

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
            if (nei < 1.0/3.0) {
                // swap
                std::swap(results.Sequence[i_st], results.Sequence[i_nd]);

            } else if (nei < 2.0/3.0) {
                // insertion
                Item holder = results.Sequence[i_nd];
                if (i_st < i_nd) {
                    for (int i = i_nd; i > i_st; --i) {
                        results.Sequence[i] = results.Sequence[i - 1];
                    }
                } else {
                    for (int i = i_nd; i < i_st; ++i) {
                    results.Sequence[i] = results.Sequence[i + 1];
                    }
                }
                results.Sequence[i_st] = holder;

            } else {
                int lo = i_st, hi = i_nd;
                while (lo < hi)
                    std::swap(results.Sequence[lo++], results.Sequence[hi--]);
            }
        }
    }

    return results;
}

float initial_temp(const Sequence_vec& Sequence, int n, int seed, int max_rows, int Lk, float ratio) {
    
    std::mt19937 g(seed);
    SAResults result = sa_iteration(Sequence, Lk, std::numeric_limits<float>::infinity(), g, n, max_rows);

    return std::abs((result.average_dif/Lk)/log(ratio));
}

struct SARuns {
    Sequence_vec Sequence_best;
    int best_cost;
    double time;
    int seed;
};

SARuns SA(Sequence_vec& Sequence, int n, int seed, int max_rows, int max_cols, float c_in, int Lk, float cooling_rate, int freeze_crit) {
    SARuns run;
    run.seed = seed;

    // Print the generated sequence
    //for (size_t i = 0; i < Sequence.size(); ++i) {
    //    printf("(%d, %d, %d),\n", Sequence[i].x, Sequence[i].y, Sequence[i].w);
    //}

    std::mt19937 g(seed);
    std::shuffle(Sequence.begin(), Sequence.end(), g);

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
        //printf("%f,%d\n",c_in, best_cost);
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
    int n = 75;
    
    // Initialize RNG
    std::mt19937 gen(seed);   // Mersenne Twister RNG
    std::uniform_int_distribution<> dist_x(0, max_cols - 1);
    std::uniform_int_distribution<> dist_y(0, max_rows - 1);


    // To avoid duplicates, use a set
    std::unordered_set<std::string> used;
    Sequence_vec Sequence;

    while (Sequence.size() < n) {
        int x = dist_x(gen);
        int y = dist_y(gen);

        std::ostringstream key;
        key << x << "," << y;

        if (used.insert(key.str()).second) {  // only add if new
            Sequence.push_back({x, y, 0});
        }
    }
    int weight_total = 0;
    std::uniform_int_distribution<> weight(1, 5);
    for (int i = 0; i < Sequence.size(); i++) {
        Sequence[i].w = weight(gen);
        weight_total += Sequence[i].w;
    }

    printf("%d, %d", weight_total, weight_total/25);

    for (int i = 0; i < (weight_total/25)-1; i++) {
        int x = 0;
        int y = 0;
        n++;

        std::ostringstream key;
        key << x << "," << y;
        Sequence.push_back({x, y, 0});
    }

    //for (int i = 0; i < Sequence.size(); i++) {
    //    printf("(%d, %d, %d),", Sequence[i].x, Sequence[i].y, Sequence[i].w);
    //}

    int Lk = 1000 * (n * (n - 1) / 2);
    int freeze_crit = 25;
    float cooling_rate = 0.8f;
    float ratio = 0.5f;
    float c_in = initial_temp(Sequence, n, seed, max_rows, Lk, ratio);
    printf("Initial Temp: %f\n", c_in);

    float cost_avg = 0;
    float time_avg = 0;

    for (int k = 0; k < 1; k++) {
        SARuns run = SA(Sequence, n, k, max_rows, max_cols, c_in, Lk, cooling_rate, freeze_crit);
        cost_avg += run.best_cost;
        time_avg += run.time;

        //for (size_t i = 0; i < run.Sequence_best.size(); ++i) {
        //    printf("(%d, %d, %d),\n", run.Sequence_best[i].x, run.Sequence_best[i].y, run.Sequence_best[i].w);
        //}
        printf("\n[%d, %.3f, %d],", run.best_cost, run.time, k);
    }
    printf("\n");
    printf("\n[%f, %f],", cost_avg/1, time_avg/1);
}