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

#define num_exams 5  // Adjust to match your dimensions
#define max_rows 20
#define max_cols 8

float rec[num_exams][max_rows][max_cols];

// Type aliases for vectors used in ini_res_usage
using rec_vec = std::vector<std::vector<std::vector<int>>>;
using RU_vec = std::vector<std::vector<int>>;
using X_vec = std::vector<std::vector<std::vector<int>>>;
using R_vec = std::vector<std::vector<std::vector<int>>>;
using S_vec = std::vector<std::vector<int>>;
using F_vec = std::vector<std::vector<int>>;
using delta_vec = std::vector<std::vector<std::vector<int>>>;
using Sequence_vec = std::vector<int>;

void ini_res_usage(RU_vec& resource_usage, const X_vec& X_t_ik, const rec_vec& rec_lpk, const int* pro_i) {
    for (int t = 0; t < (int)resource_usage.size(); t++) {
        for (int p = 0; p < max_rows-2; p++) {
            int sum = 0;
            for (int i = 0; i < (int)X_t_ik[t].size(); i++) {
                for (int k = 0; k < max_cols; k++) {
                    sum += X_t_ik[t][i][k] * rec_lpk[pro_i[i]][p+2][k];
                }
            }
            resource_usage[t][p] = sum;
        }
    }
}

void precompute_exam_type_patterns(delta_vec& delta, const RU_vec& resource_usage, const S_vec& S_ik) {
    for (int exam = 0; exam < (int)delta.size(); exam++) {
        int start = S_ik[exam][0];
        for (int t = 0; t < (int)delta[exam].size(); t++) {
            for (int p = 0; p < (int)delta[exam][t].size(); p++) {
                delta[exam][t][p] = resource_usage[start + t][p];
            }
        }
    }
}

void load_data(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    size_t elements_read = fread(rec, sizeof(float), num_exams * max_rows * max_cols, file);
    if (elements_read != (size_t)(num_exams * max_rows * max_cols)) {
        perror("Error reading file");
        exit(1);
    }

    fclose(file);
}

int earliest_candidate(RU_vec& RU_define_sequence, const delta_vec& delta, const int* rec_max_p, const int* durations, int i, int start, int end, const std::vector<int>& exam_type_start) {
    int pri1 = 8;
    int pri2 = 16;
    const auto& delta_exam = delta[i];
    const int duration = durations[i];
    if (start < exam_type_start[i]) {
        start = exam_type_start[i];
    }
    for (int t = start; t <= end - duration; t++) {
        bool valid = true;
        for (int t_prime = 0; t_prime < duration; t_prime++) {
            const auto& delta_time = delta_exam[t_prime];
            if (delta_time[pri1] != 0 || delta_time[pri2] != 0) {
                if (RU_define_sequence[t + t_prime][pri1] + delta_time[pri1] > rec_max_p[pri1] ||
                    RU_define_sequence[t + t_prime][pri2] + delta_time[pri2] > rec_max_p[pri2]) {
                    valid = false;
                    break;
                }
            }
        }
        if (!valid) continue;

        for (int t_prime = 0; t_prime < duration; t_prime++) {
            const auto& delta_time = delta_exam[t_prime];
            const auto& RU = RU_define_sequence[t + t_prime];
            for (int p = 0; p < pri1; p++) {
                if (delta_time[p] != 0) {
                    if (RU[p] + delta_time[p] > rec_max_p[p + 2]) {
                        valid = false;
                        break;
                    }
                }
            }
            if (!valid) break;
            for (int p = pri1+1; p < pri2; p++) {
                if (delta_time[p] != 0) {
                    if (RU[p] + delta_time[p] > rec_max_p[p + 2]) {
                        valid = false;
                        break;
                    }
                }
            }
            if (!valid) break;
            for (int p = pri2+1; p < max_rows-2; p++) {
                if (delta_time[p] != 0) {
                    if (RU[p] + delta_time[p] > rec_max_p[p + 2]) {
                        valid = false;
                        break;
                    }
                }
            }
            if (!valid) break;
        }
        if (valid) return t;
    }
    return -1;
}

std::vector<int> define_sequence_LS(const Sequence_vec& Sequence, const delta_vec& delta, RU_vec& RU_define_sequence, int T, int n, const int* rec_max_p, const int* pro_i, const int* durations, int start, int end) {
    std::vector<int> S(n, 0);
    std::vector<int> exam_type_start(num_exams, 0);
    for (int i = 0; i < n; i++) {
        int exam = Sequence[i];
        int exam_type = pro_i[exam];
        const auto& delta_exam = delta[exam_type];
        if (exam_type_start[exam_type] <= end) {
            int t = earliest_candidate(RU_define_sequence, delta, rec_max_p, durations, exam_type, start, end, exam_type_start);
            if (t == -1) {
                t = T - durations[exam_type] - 1;
            } else {
                for (int t_prime = 0; t_prime < durations[exam_type]; t_prime++) {
                    const auto& delta_time = delta_exam[t_prime];
                    for (int p = 0; p < max_rows-2; p++) {
                        RU_define_sequence[t + t_prime][p] += delta_time[p];
                    }
                }
            }
            if (t > exam_type_start[exam_type]) {
                exam_type_start[exam_type] = t;
            }

            S[exam] = t;
        } else {
            S[exam] = T - durations[exam_type] - 1;
        }
    }
    return S;
}

int compute_cost(const std::vector<int>& S, const int* durations, int n, const int* pro_i) {
    int cost = 0;
    for (int i = 0; i < n; i++) {
        if (S[i] + durations[pro_i[i]] <= 487) {
            cost--;
        }
    }
    return cost;
}

struct SAResults {
    int best_cost;
    Sequence_vec Sequence_best;
    Sequence_vec Sequence;
    float average_dif;
};

SAResults sa_iteration(const delta_vec& delta, int T, int n, const int* rec_max_p, const int* pro_i, const int* durations, int Lk, float c, const Sequence_vec& initial_sequence, std::mt19937& g) {
    SAResults results;
    results.Sequence = initial_sequence;          // current solution
    results.Sequence_best = initial_sequence;     // best found so far
    results.average_dif = 0.0f;

    // compute initial schedule and cost
    RU_vec RU_define_sequence(T, std::vector<int>(max_rows-2, 0));
    std::vector<int> S_best = define_sequence_LS(results.Sequence_best, delta, RU_define_sequence, T, n, rec_max_p, pro_i, durations, 0, 487);
    results.best_cost = compute_cost(S_best, durations, n, pro_i);
    int current_cost = results.best_cost;

    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    for (int it = 0; it < Lk; it++) {
        int i_st = g() % n;
        int i_nd = g() % n;
        // ensure different positions and different exam types (as original intent)
        while ((i_st == i_nd || pro_i[results.Sequence[i_st]] == pro_i[results.Sequence[i_nd]])) {
            i_st = g() % n;
            i_nd = g() % n;
        }

        std::swap(results.Sequence[i_st], results.Sequence[i_nd]);

        RU_define_sequence.assign(T, std::vector<int>(max_rows-2, 0));
        auto S = define_sequence_LS(results.Sequence, delta, RU_define_sequence, T, n, rec_max_p, pro_i, durations, 0, 487);
        int alt_cost = compute_cost(S, durations, n, pro_i);

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
            if (current_cost < results.best_cost) {
                results.best_cost = current_cost;
                results.Sequence_best = results.Sequence;
            }
        } else {
            // revert swap
            std::swap(results.Sequence[i_st], results.Sequence[i_nd]);
        }
    }

    // average_dif over iterations to be meaningful:
    if (Lk > 0) results.average_dif /= (float)Lk;

    return results;
}

struct SARuns {
    Sequence_vec Sequence_best;
    int best_cost_ever = 1000000;
    double time;
    int seed;
};

SARuns SA(const delta_vec& delta, const int* rec_max_p, const int* pro_i, const int* durations, int T, int n, int seed, float c_in, int Lk, float cooling_rate, int freeze_crit) {
    SARuns run;

    Sequence_vec Sequence(n);
    for (int i = 0; i < n; i++) {
        Sequence[i] = i;
    }

    std::mt19937 g(seed);   // Mersenne Twister RNG
    std::shuffle(Sequence.begin(), Sequence.end(), g);

    int freeze = freeze_crit;
    Sequence_vec alt_time_best_Sequence(n);
        
    RU_vec RU_define_sequence(T, std::vector<int>(max_rows-2, 0));
    auto t1 = std::chrono::high_resolution_clock::now();
    while (freeze > 0) {
        SAResults result = sa_iteration(delta, T, n, rec_max_p, pro_i, durations, Lk, c_in, Sequence, g);
        RU_define_sequence.assign(T, std::vector<int>(max_rows-2, 0));
        int alt_cost = result.best_cost;
        Sequence = result.Sequence;

        if (alt_cost < run.best_cost_ever) {
            freeze = freeze_crit;
            run.best_cost_ever = alt_cost;
            alt_time_best_Sequence = result.Sequence_best;

        } else {
            freeze--;
        }
        c_in = c_in * cooling_rate;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;

    run.time = elapsed.count();
    run.Sequence_best = alt_time_best_Sequence;

    return run;
}

int main() {
    int seed = 50;
    int pro_i[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    int n = sizeof(pro_i)/sizeof(pro_i[0]);

    load_data("rec_lpk.bin");
    printf("Data loaded successfully!\n");
    int rec_max_p[] = {0, 0, 4, 4, 4, 2, 1, 2, 1, 4, 1, 2, 3, 1, 1, 2, 1, 6, 1, 3};

    rec_vec rec_lpk(num_exams, std::vector<std::vector<int>>(max_rows, std::vector<int>(max_cols, 0)));
    for (int i = 0; i < num_exams; i++) {
        for (int j = 0; j < max_rows; j++) {
            for (int k = 0; k < max_cols; k++) {
                rec_lpk[i][j][k] = (int)rec[i][j][k];
            }
        }
    }

    int T = 0;
    for (int i = 0; i < n; i++) {
        int exam = pro_i[i];
        for (int k = 0; k < max_cols; k++) {
            T = T + rec_lpk[exam][1][k];
        }
    }

    X_vec X_t_ik(T, std::vector<std::vector<int>>(n, std::vector<int>(max_cols, 0)));
    S_vec S_ik(n, std::vector<int>(max_cols, 0));
    F_vec F_ik(n, std::vector<int>(max_cols, 0));
    int durations[num_exams];

    int exam_type_size[num_exams];
    for (int i = 0; i < num_exams; i++) {
        exam_type_size[i] = 0;
        for (int j = 0; j < n; j++) {
            if (pro_i[j] == i) {
                exam_type_size[i]++;
            }
        }
    }

    // precompute exam patterns for each exam type
    int t = 0;
    int real_i_index = 0;
    for (int i = 0; i < num_exams; i++) {
        S_ik[i][0] = t;
        F_ik[i][0] = S_ik[i][0] + rec_lpk[i][1][0];
        for (int t_prime = S_ik[i][0]; t_prime < F_ik[i][0]; t_prime++) {
            X_t_ik[t_prime][real_i_index][0] = 1;
        }
        for (int k = 1; k < max_cols; k++) {
            S_ik[i][k] = F_ik[i][k-1];
            F_ik[i][k] = S_ik[i][k] + rec_lpk[i][1][k];
            for (int t_prime = S_ik[i][k]; t_prime < F_ik[i][k]; t_prime++) {
                X_t_ik[t_prime][real_i_index][k] = 1;
            }
        }
        real_i_index = real_i_index + exam_type_size[i];
        durations[i] = F_ik[i][max_cols-1] - S_ik[i][0];
        t = F_ik[i][max_cols-1];
    }

    delta_vec delta(num_exams);
    for (int i = 0; i < num_exams; i++) {
        delta[i].resize(durations[i], std::vector<int>(max_rows-2, 0));
    }

    RU_vec resource_usage(T, std::vector<int>(max_rows-2, 0));
    ini_res_usage(resource_usage, X_t_ik, rec_lpk, pro_i);
    precompute_exam_type_patterns(delta, resource_usage, S_ik);

    int n_runs = 100;
    int n_threads = 10;
    float c_in = 0.9767f;
    int freeze_crit = 55;
    int Lk = 250;
    float cooling_rate = 0.975f;

    // Results container
    std::vector<SARuns> results(n_runs);

    // Mutex for printing
    std::mutex cout_mtx;

    // Containers to collect per-batch (per-set-of-n_threads) statistics
    std::vector<int> batch_min_costs;
    std::vector<double> batch_max_times;

    // Run in batches of up to n_threads
    for (int batch_start = 0; batch_start < n_runs; batch_start += (int)n_threads) {
        int batch_end = std::min(n_runs, batch_start + (int)n_threads);
        std::vector<std::thread> threads;
        threads.reserve(batch_end - batch_start);

        for (int run_idx = batch_start; run_idx < batch_end; ++run_idx) {
            int seed = run_idx;
            threads.emplace_back([&, run_idx, seed]() {
            // Each thread calls SA with its own seed. All other inputs are read-only.
            SARuns r = SA(delta, rec_max_p, pro_i, durations, T, n, seed, c_in, Lk, cooling_rate, freeze_crit);
            results[run_idx] = r;

            std::lock_guard<std::mutex> lg(cout_mtx);
            printf("[Thread] run %d finished: seed=%d best_cost=%d time=%.3f s\n", run_idx, seed, r.best_cost_ever, r.time);
            });
        }

        // join this batch
        for (auto &th : threads) {
            if (th.joinable()) th.join();
        }

        // compute batch-level stats: lowest cost and highest time among runs in this batch
        int batch_min_cost = 1000000000; // large sentinel
        double batch_max_time = 0.0;
        for (int i = batch_start; i < batch_end; ++i) {
            // Ensure the result for this index was filled -- if SA can fail, check appropriately
            batch_min_cost = std::min(batch_min_cost, results[i].best_cost_ever);
            batch_max_time = std::max(batch_max_time, results[i].time);
        }
        batch_min_costs.push_back(batch_min_cost);
        batch_max_times.push_back(batch_max_time);

        std::lock_guard<std::mutex> lg(cout_mtx);
        printf("Batch %d..%d -> min_cost=%d, max_time=%.3f s\n\n", batch_start, batch_end-1, batch_min_cost, batch_max_time);
    }

    // Print summary for all batches
    printf("Summary per batch (each batch contains up to %u runs):\n", n_threads);
    for (size_t b = 0; b < batch_min_costs.size(); ++b) {
        printf("Batch %02zu: min_cost=%d, max_time=%.3f s\n", b, batch_min_costs[b], batch_max_times[b]);
    }

    // Compute and print averages of the batch min_costs and batch max_times
    if (!batch_min_costs.empty()) {
        double sum_min_costs = 0.0;
        double sum_max_times = 0.0;
        for (size_t b = 0; b < batch_min_costs.size(); ++b) {
            sum_min_costs += static_cast<double>(batch_min_costs[b]);
            sum_max_times += batch_max_times[b];
        }
        double avg_min_cost = sum_min_costs / static_cast<double>(batch_min_costs.size());
        double avg_max_time = sum_max_times / static_cast<double>(batch_max_times.size());
        printf("Averages over %zu batches: avg(min_cost)=%.3f, avg(max_time)=%.3f s\n", batch_min_costs.size(), avg_min_cost, avg_max_time);
    } else {
        printf("No batches were run; cannot compute averages.\n");
    }

    // Find overall best result among all runs (unchanged)
    int best_idx = 0;
    int best_cost = results[0].best_cost_ever;
    for (int i = 1; i < n_runs; ++i) {
        if (results[i].best_cost_ever < best_cost) {
            best_cost = results[i].best_cost_ever;
            best_idx = i;
        }
    }

    printf("Overall best run: %d with cost %d (time %.3f s)\n",best_idx, results[best_idx].best_cost_ever, results[best_idx].time);

    return 0;
}
