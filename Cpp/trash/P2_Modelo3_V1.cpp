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

#define num_exams 5  // Adjust to match your dimensions
#define max_rows 20
#define max_cols 8

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

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
    int best_cost = compute_cost(S_best, durations, n, pro_i);
    int current_cost = best_cost;

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
            if (current_cost < best_cost) {
                best_cost = current_cost;
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

    Sequence_vec Sequence(n);
    for (int i = 0; i < n; i++) {
        Sequence[i] = i;
    }

    std::mt19937 g(seed);   // Mersenne Twister RNG
    std::shuffle(Sequence.begin(), Sequence.end(), g);

    float c_in = 100.0f;
    int freeze_crit = 10;
    int Lk = 10000;
    float cooling_rate = 0.9f;
    int freeze = freeze_crit;
    Sequence_vec alt_time_best_Sequence(n);

    // Call SA with initial shuffled sequence and RNG
    RU_vec RU_define_sequence(T, std::vector<int>(max_rows-2, 0));
    int best_cost = T;
    auto t1 = std::chrono::high_resolution_clock::now();
    while (freeze > 0) {
        SAResults result = sa_iteration(delta, T, n, rec_max_p, pro_i, durations, Lk, c_in, Sequence, g);
        RU_define_sequence.assign(T, std::vector<int>(max_rows-2, 0));
        auto S = define_sequence_LS(result.Sequence_best, delta, RU_define_sequence, T, n, rec_max_p, pro_i, durations, 0, 487);
        int alt_cost = compute_cost(S, durations, n, pro_i);
        Sequence = result.Sequence;

        if (alt_cost < best_cost) {
            freeze = freeze_crit;
            best_cost = alt_cost;
            alt_time_best_Sequence = result.Sequence_best;

        } else {
            freeze--;
        }
        c_in = c_in * cooling_rate;
        printf("Current c_in: %f\n", c_in);
        printf("Current cost: %d\n", best_cost);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    printf("Elapsed time: %.2f seconds\n", elapsed.count());
    printf("Best sequence:\n[");
    for (int i = 0; i < n; ++i) {
        printf("%d,", alt_time_best_Sequence[i]);
    }
    printf("]\n");

    RU_vec RU_define_sequence_final(T, std::vector<int>(max_rows-2, 0));
    // RU_define_sequence.assign(T, std::vector<int>(max_rows-2, 0)); Works slower? Strange... 31s vs 24.5
    std::vector<int> S_best = define_sequence_LS(alt_time_best_Sequence, delta, RU_define_sequence_final, T, n, rec_max_p, pro_i, durations, 0, 487);
    best_cost = compute_cost(S_best, durations, n, pro_i);
    for (int i = 0; i < n; ++i) {
        printf("%d,", S_best[i]);
    }
    printf("]\n");
    printf("\nBest makespan: %d\n", best_cost);

    double sum = 0;
    for (int times = 0; times < 1000; times++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        S_best = define_sequence_LS(alt_time_best_Sequence, delta, RU_define_sequence_final, T, n, rec_max_p, pro_i, durations, 0, 487);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t2 - t1;
        sum = sum + diff.count();
    }
    std::cout << "define_sequence_LS took " << sum/1000 << " s\n";

    return 0;
}
