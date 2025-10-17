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

int earliest_candidate(RU_vec& RU_define_sequence, const delta_vec& delta, const int* rec_max_p, int i, int start, int end) {
    int pri1 = 8;
    int pri2 = 16;
    for (int t = start; t <= end - (int)delta[i].size(); t++) {
        bool valid = true;
        for (int t_prime = 0; t_prime < (int)delta[i].size(); t_prime++) {
            int v1 = delta[i][t_prime][pri1];
            if (RU_define_sequence[t + t_prime][pri1] + v1 > rec_max_p[pri1]) {
                valid = false;
                break;
            }
        }
        for (int t_prime = 0; t_prime < (int)delta[i].size(); t_prime++) {
            int v2 = delta[i][t_prime][pri2];
            if (RU_define_sequence[t + t_prime][pri2] + v2 > rec_max_p[pri2]) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;

        for (int t_prime = 0; t_prime < (int)delta[i].size(); t_prime++) {
            for (int p = 0; p < max_rows - 2; p++) {
                if (p == pri1 || p == pri2) continue;
                if (RU_define_sequence[t + t_prime][p] + delta[i][t_prime][p] > rec_max_p[p + 2]) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }
        if (valid) return t;
    }
    return -1;
}

std::vector<int> define_sequence_LS(const Sequence_vec& Sequence, const delta_vec& delta, int T, int n, const int* rec_max_p, const int* pro_i) {
    RU_vec RU_define_sequence(T, std::vector<int>(max_rows-2, 0));
    std::vector<int> S(n, 0);
    for (int i = 0; i < n; i++) {
        int exam = Sequence[i];
        int exam_type = pro_i[exam];
        int t = earliest_candidate(RU_define_sequence, delta, rec_max_p, exam_type, 0, T);
        if (t < 0) {
            // No feasible slot found; store -1 or handle as you wish
            S[exam] = -1;
            continue;
        }
        for (int t_prime = 0; t_prime < (int)delta[exam_type].size(); t_prime++) {
            for (int p = 0; p < max_rows-2; p++) {
                RU_define_sequence[t + t_prime][p] += delta[exam_type][t_prime][p];
            }
        }
        S[exam] = t;
    }
    return S;
}

int compute_cost(const std::vector<int>& S, const delta_vec& delta, int n, const int* pro_i) {
    int cost = 0;
    for (int i = 0; i < n; i++) {
        if (S[i] >= 0) {
            int finish = S[i] + (int)delta[pro_i[i]].size();
            if (finish > cost) cost = finish;
        }
    }
    return cost;
}

struct SAResults {
    Sequence_vec Sequence_best;
    Sequence_vec Sequence;
    float average_dif;
};

SAResults sa_iteration(const delta_vec& delta, int T, int n, const int* rec_max_p, const int* pro_i, int Lk, float c, const Sequence_vec& initial_sequence, std::mt19937& g) {
    SAResults results;
    results.Sequence = initial_sequence;          // current solution
    results.Sequence_best = initial_sequence;     // best found so far
    results.average_dif = 0.0f;

    // compute initial schedule and cost
    std::vector<int> S_best = define_sequence_LS(results.Sequence_best, delta, T, n, rec_max_p, pro_i);
    int best_cost = compute_cost(S_best, delta, n, pro_i);
    int current_cost = best_cost;

    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    for (int it = 0; it < Lk; it++) {
        int i_st = g() % n;
        int i_nd = g() % n;
        // ensure different positions and different exam types (as original intent)
        int tries = 0;
        while ((i_st == i_nd || pro_i[results.Sequence[i_st]] == pro_i[results.Sequence[i_nd]]) && tries < 1000) {
            i_st = g() % n;
            i_nd = g() % n;
            tries++;
        }

        std::swap(results.Sequence[i_st], results.Sequence[i_nd]);

        auto S = define_sequence_LS(results.Sequence, delta, T, n, rec_max_p, pro_i);
        int alt_cost = compute_cost(S, delta, n, pro_i);

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
    int pro_i[] = {0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
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

    // Construct a new schedule and compute its resource usage
    X_t_ik.assign(T, std::vector<std::vector<int>>(n, std::vector<int>(max_cols, 0)));
    S_ik.assign(n, std::vector<int>(max_cols, 0));
    F_ik.assign(n, std::vector<int>(max_cols, 0));
    resource_usage.assign(T, std::vector<int>(max_rows-2, 0));

    t = 0;
    for (int i = 0; i < n; i++) {
        S_ik[i][0] = t;
        F_ik[i][0] = S_ik[i][0] + rec_lpk[pro_i[i]][1][0];
        for (int k = 1; k < max_cols; k++) {
            S_ik[i][k] = F_ik[i][k-1];
            F_ik[i][k] = S_ik[i][k] + rec_lpk[pro_i[i]][1][k];
        }
        for (int t_prime = 0; t_prime < (int)delta[pro_i[i]].size(); t_prime++) {
            for (int p = 0; p < max_rows-2; p++) {
                resource_usage[S_ik[i][0] + t_prime][p] += delta[pro_i[i]][t_prime][p];
            }
        }
        t = F_ik[i][max_cols-1];
    }

    float c_in = 1.0f;

    Sequence_vec Sequence(n);
    for (int i = 0; i < n; i++) {
        Sequence[i] = i;
    }

    std::mt19937 g(seed);   // Mersenne Twister RNG
    std::shuffle(Sequence.begin(), Sequence.end(), g);

    // Call SA with initial shuffled sequence and RNG
    SAResults result = sa_iteration(delta, T, n, rec_max_p, pro_i, 1000, c_in, Sequence, g);

    printf("Best sequence:\n[");
    for (int i = 0; i < n; ++i) {
        printf("%d,", result.Sequence_best[i]);
    }
    printf("]\n");

    std::vector<int> S_best = define_sequence_LS(result.Sequence_best, delta, T, n, rec_max_p, pro_i);
    int best_cost = compute_cost(S_best, delta, n, pro_i);
    for (int i = 0; i < n; ++i) {
        printf("%d,", S_best[i]);
    }
    printf("]\n");
    printf("\nBest makespan: %d\n", best_cost);
    printf("\nAverage diff: %f\n", result.average_dif);

    double sum = 0;
    for (int times = 0; times < 100; times++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        S_best = define_sequence_LS(result.Sequence_best, delta, T, n, rec_max_p, pro_i);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t2 - t1;
        sum = sum + diff.count();
    }
    std::cout << "define_sequence_LS took " << sum/100 << " s\n";

    return 0;
}
