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

#define num_exams 5  // Adjust to match your dimensions
#define max_rows 20
#define max_cols 8

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

float rec_lpk[num_exams][max_rows][max_cols];

// Type aliases for vectors used in ini_res_usage
using RU_vec = std::vector<std::vector<int>>;
using X_vec = std::vector<std::vector<std::vector<int>>>;
using R_vec = std::vector<std::vector<std::vector<int>>>;
using S_vec = std::vector<std::vector<int>>;
using F_vec = std::vector<std::vector<int>>;
using delta_vec = std::vector<std::vector<std::vector<int>>>;
using Sequence_vec = std::vector<int>;

void ini_res_usage(RU_vec& resource_usage, const X_vec& X_t_ik, const R_vec& R, const int* pro_i) {
    int sum = 0;
    // Update indexing to use integers
    for (int t = 0; t < resource_usage.size(); t++) {
        for (int p = 0; p < resource_usage[t].size(); p++) {
            int sum = 0;
            for (int i = 0; i < X_t_ik[t].size(); i++) {
                for (int k = 0; k < max_cols; k++) {
                    sum += X_t_ik[t][i][k] * R[i][p][k];
                }
            }
            resource_usage[t][p] = sum;
        }
    }
}

void precompute_exam_type_patterns(delta_vec& delta, const RU_vec& resource_usage, const S_vec& S_ik) {
    int start = 0;
    for (int exam = 0; exam < delta.size(); exam++) {
        start = S_ik[exam][0];
        for (int t = 0; t < delta[exam].size(); t++) {
            for (int p = 0; p < delta[exam][t].size(); p++) {
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

    size_t elements_read = fread(rec_lpk, sizeof(float), num_exams * max_rows * max_cols, file);
    if (elements_read != num_exams * max_rows * max_cols) {
        perror("Error reading file");
        exit(1);
    }

    fclose(file);
}

int earliest_candidate(RU_vec& RU_define_sequence, const delta_vec& delta, const int* rec_max_p, int i, int start, int end) {
    int start_time = 0;
    bool valid;
    for (int t = start; t < end - delta[i].size(); t++) {
        start_time = t;
        valid = true;
        for (int t_prime = 0; t_prime < delta[i].size(); t_prime++) {
            for (int p = 0; p < max_rows - 2; p++) {
                if (RU_define_sequence[t + t_prime][p] + delta[i][t_prime][p] > rec_max_p[p + 2]) {
                    valid = false;
                }
            }
        }
        if (valid == true) {
            return start_time;
        } 
    }
    return -1;
}

void define_sequence_LS(const Sequence_vec& Sequence, const delta_vec& delta, int T, int n, const int* rec_max_p, const int* pro_i) {
    RU_vec RU_define_sequence(T, std::vector<int>(max_rows-2, 0));
    int t;
    for (int i = 0; i < n; i++) {
        int exam = Sequence[i];
        int exam_type = pro_i[exam];
        printf("Index %d is exam %d of type %d; \t", i, exam, exam_type);
        t = earliest_candidate(RU_define_sequence, delta, rec_max_p, exam_type, 0, T);
        printf("Scheduled exam %d at time %d\n", exam, t);
        for (int t_prime = 0; t_prime < delta[exam_type].size(); t_prime++) {
            for (int p = 0; p < max_rows-2; p++)
            RU_define_sequence[t + t_prime][p] = delta[exam_type][t_prime][p] + RU_define_sequence[t + t_prime][p];
        }
    }
}

int main() {
    int seed = 50;
    int pro_i[] = {0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    int n = sizeof(pro_i)/sizeof(pro_i[0]);

    load_data("rec_lpk.bin");
    printf("Data loaded successfully!\n");
    int rec_max_p[] = {0, 0, 4, 4, 4, 2, 1, 2, 1, 4, 1, 2, 3, 1, 1, 2, 1, 6, 1, 3};

    int T = 0;
    for (int i = 0; i < n; i++) {
        int exam = pro_i[i];
        for (int k = 0; k < max_cols; k++) {
            T = T + rec_lpk[exam][1][k];
        }  
    }

    R_vec R(n, std::vector<std::vector<int>>(max_rows-2, std::vector<int>(max_cols, 0)));

    for (int i = 0; i < n; i++) {
        for (int p = 2; p < max_rows; p++) {
            for (int k = 0; k < max_cols; k++) {
                R[i][p-2][k] = rec_lpk[i][p][k];
            }
        }
    }

    X_vec X_t_ik(T, std::vector<std::vector<int>>(n, std::vector<int>(max_cols, 0)));
    S_vec S_ik(n, std::vector<int>(max_cols, 0));
    F_vec F_ik(n, std::vector<int>(max_cols, 0));
    int durations[num_exams];

    // precompute exam patterns for each exam
    int t = 0;
    for (int i = 0; i < num_exams; i++) {
        S_ik[i][0] = t;
        F_ik[i][0] = S_ik[i][0] + rec_lpk[i][1][0];
        for (int t_prime = S_ik[i][0]; t_prime < F_ik[i][0]; t_prime++) {
            X_t_ik[t_prime][i][0] = 1;
        }
        
        for (int k = 1; k < max_cols; k++) {
            S_ik[i][k] = F_ik[i][k-1];
            F_ik[i][k] = S_ik[i][k] + rec_lpk[i][1][k];
            for (int t_prime = S_ik[i][k]; t_prime < F_ik[i][k]; t_prime++) {
                X_t_ik[t_prime][i][k] = 1;
            }
        }
        durations[i] = F_ik[i][max_cols-1] - S_ik[i][0];
        t = F_ik[i][max_cols-1];
    }
    
    delta_vec delta(num_exams);
    for (int i = 0; i < num_exams; i++) {
        delta[i].resize(durations[i], std::vector<int>(max_rows-2, 0));
    }

    RU_vec resource_usage(T, std::vector<int>(max_rows-2, 0));
    ini_res_usage(resource_usage, X_t_ik, R, pro_i);
    precompute_exam_type_patterns(delta, resource_usage, S_ik);

    int i = 2;
    printf("%ld\n", delta[i].size());
    printf("[\n");
    for (int t = 0; t < delta[i].size(); t++) {
        printf("[");
        for (int p = 0; p < delta[i][t].size(); p++) {
            printf("%d,",delta[i][t][p]);
        }
        printf("]\n");
    }
    printf("]\n");

    i = 9;
    printf("[\n");
    for (int p = 0; p < max_rows-2; p++) {
        printf("[");
        for (int k = 0; k < max_cols; k++) {
            printf("%d,",R[i][p][k]);
        }
        printf("]\n");
    }
    printf("]\n");

    // Consttruct a new schedule and compute its resource usage
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
        for (int t_prime = 0; t_prime < delta[pro_i[i]].size(); t_prime++) {
            for (int p = 0; p < max_rows-2; p++)
            resource_usage[S_ik[i][0] + t_prime][p] = delta[pro_i[i]][t_prime][p] + resource_usage[S_ik[i][0] + t_prime][p];
        }
        t = F_ik[i][max_cols-1];
    }

    float c_in = 5.0;

    Sequence_vec Sequence(n, 0);
    for (int i = 0; i < n; i++) {
        Sequence[i] = i;
    }

    std::mt19937 g(seed);   // Mersenne Twister RNG

    // Shuffle the vector
    std::shuffle(Sequence.begin(), Sequence.end(), g);

    for (int i = 0; i < n; i++) {
        printf("Shuffled sequence is: %d\n", Sequence[i]);
    }

    define_sequence_LS(Sequence, delta, T, n, rec_max_p, pro_i);

    return 0;
}