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
    int x, y, w, idx;
};

using Sequence_vec = std::vector<Item>;
using coord = std::vector<int>;
using distances_flat = std::vector<int>;

inline int distance_func(int x1, int y1, int x2, int y2, int max_rows) {
    int v_dist;
    int h_dist = std::abs(x1 - x2);

    int sum = y1 + y2;
    int mirror = (2*max_rows) - sum;

    if (h_dist != 0) {
        v_dist = mirror > sum? sum : mirror;

    } else {
        v_dist = std::abs(y1 - y2);
    }

    return h_dist + v_dist;
}

int cost_func(const Sequence_vec& Sequence, const distances_flat& dist_array, int n, int max_rows) {
    int dist = 0;
    int max_storage = 25;
    int current_storage = Sequence[0].w;;
    int c_x = 0, y_x = 0;
    int idx1, idx2;
    int over_wei = 0;
    
    dist += dist_array[(n-1)*n + Sequence[0].idx];
    idx1 = Sequence[0].idx;
    
    for (int i = 1; i < n; i++) {
        idx2 = Sequence[i].idx;

        if (Sequence[i].x == c_x && Sequence[i].y == y_x) {
            current_storage = 0;
            over_wei = 0;
        }else {
            current_storage += Sequence[i].w;
            if (current_storage > max_storage) {
                over_wei = 10*(current_storage - max_storage);
            }
        }
        dist += dist_array[(idx1 * n) + idx2] + over_wei;
        idx1 = idx2;

    }
    dist += dist_array[(n-1)*n + Sequence[n-1].idx];

    return dist;
}

void calc_dist_array(const Sequence_vec& Sequence, distances_flat& dist_array, int n, int max_rows) {
    for (int i = 0; i < n; ++i) {
        int x1 = Sequence[i].x, y1 = Sequence[i].y;
        for (int j = i + 1; j < n; ++j) {
            int d = distance_func(x1, y1, Sequence[j].x, Sequence[j].y, max_rows);
            dist_array[i * n + j] = d;
            dist_array[j * n + i] = d;
        }
    }
}

struct SAResults {
    Sequence_vec Sequence_best;
    Sequence_vec Sequence;
    float average_dif;
};

SAResults sa_iteration(const Sequence_vec& initial_sequence, distances_flat& dist_array, int Lk, float c, std::mt19937& g, int n, int max_rows) {
    SAResults results;
    results.Sequence = initial_sequence;          // current solution
    results.Sequence_best = initial_sequence;     // best found so far
    results.average_dif = 0.0f;

    int best_cost = cost_func(results.Sequence, dist_array, n, max_rows);
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

        int alt_cost = cost_func(results.Sequence, dist_array, n, max_rows);

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

float initial_temp(const Sequence_vec& Sequence, distances_flat& dist_array, int n, int seed, int max_rows, int Lk, float ratio) {

    std::mt19937 g(seed);
    SAResults result = sa_iteration(Sequence, dist_array, Lk, std::numeric_limits<float>::infinity(), g, n, max_rows);

    return std::abs((result.average_dif/Lk)/log(ratio));
}

struct SARuns {
    Sequence_vec Sequence_best;
    int best_cost;
    double time;
    int seed;
};

SARuns SA(Sequence_vec& Sequence, distances_flat& dist_array, int n, int seed, int max_rows, int max_cols, float c_in, int Lk, float cooling_rate, int freeze_crit) {
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
        
    int best_cost = cost_func(Sequence, dist_array, n, max_rows);
    auto t1 = std::chrono::high_resolution_clock::now();
    while (freeze > 0) {
        SAResults result = sa_iteration(Sequence, dist_array, Lk, c_in, g, n, max_rows);
        int alt_cost = cost_func(Sequence, dist_array, n, max_rows);

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
            Sequence.push_back({x, y, 0, 0});
        }
    }
    int weight_total = 0;
    std::uniform_int_distribution<> weight(1, 5);
    for (int i = 0; i < Sequence.size(); i++) {
        Sequence[i].w = weight(gen);
        weight_total += Sequence[i].w;
    }

    for (int i = 0; i < Sequence.size(); i++) {
        Sequence[i].idx = i;
    }

    printf("%d, %d", weight_total, weight_total/25);

    int final_n = n;
    for (int i = 0; i < (weight_total/25)-1; i++) {
        int x = 0;
        int y = 0;
        int w = 0;
        n++;

        std::ostringstream key;
        key << x << "," << y;
        Sequence.push_back({x, y, 0, final_n});
    }

    for (int i = 0; i < Sequence.size(); i++) {
        printf("(%d, %d, %d, %d),", Sequence[i].x, Sequence[i].y, Sequence[i].w, Sequence[i].idx);
    }

    distances_flat dist_array(n * n);
    calc_dist_array(Sequence, dist_array, n, max_rows);

    int Lk = 1000 * (n * (n - 1) / 2);
    int freeze_crit = 25;
    float cooling_rate = 0.8f;
    float ratio = 0.5f;
    float c_in = initial_temp(Sequence, dist_array, n, seed, max_rows, Lk, ratio);
    printf("Initial Temp: %f\n", c_in);

    float cost_avg = 0;
    float time_avg = 0;

    for (int k = 0; k < 1; k++) {
        SARuns run = SA(Sequence, dist_array, n, k, max_rows, max_cols, c_in, Lk, cooling_rate, freeze_crit);
        cost_avg += run.best_cost;
        time_avg += run.time;

        for (size_t i = 0; i < run.Sequence_best.size(); ++i) {
            printf("(%d, %d, %d, %d),\n", run.Sequence_best[i].x, run.Sequence_best[i].y, run.Sequence_best[i].w, run.Sequence_best[i].idx);
        }
        printf("\n[%d, %.3f, %d],", run.best_cost, run.time, k);
    }
    printf("\n");
    printf("\n[%f, %f],", cost_avg/1, time_avg/1);
}