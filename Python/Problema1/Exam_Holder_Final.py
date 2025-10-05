# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:48:38 2025

@author: JAPGc
"""

import numpy as np

def pad_table(table, max_cols):
    # Pads a table with extra columns if needed.
    current_cols = table.shape[1]
    if current_cols < max_cols:
        extra = max_cols - current_cols
        first_row = table[0]
        start_val = first_row[-1] + 1 if first_row.size > 0 else 0
        pad_first = np.arange(start_val, start_val + extra)
        new_first_row = np.concatenate([first_row, pad_first])
        new_other_rows = [
            np.concatenate([row, np.zeros(extra, dtype=table.dtype)]) for row in table[1:]
        ]
        padded_table = np.vstack([new_first_row, np.array(new_other_rows)])
    else:
        padded_table = table.copy()
    return padded_table


def exams_to_assign(table):
    # Determine maximum dimensions and pad each rec array accordingly
    max_cols = max(t.shape[1] for t in table)
    print(max_cols)
    rec_lpk = np.array([pad_table(t, max_cols) for t in table])

    # Save the resulting array to a .npy file
    np.save('rec_lpk.npy', rec_lpk)
    print("Saved rec_lpk.npy successfully.")

# Define your rec arrays (you can add more here as needed)

#Cintigrafia Miocárdica de perfusão em esforço/stress farmacológico
rec0_aut = np.array([
    [0,  1, 2, 3, 4, 5, 6, 7], # task
    [15, 8,13,60,15, 4, 3, 5], # duration
    [0,  0, 0, 1, 1, 1, 0, 0], # RH TAS
    [0,  1, 1, 0, 0, 0, 1, 1], # RH Enf
    [0,  0, 0, 0, 1, 0, 0, 0], # RH Tec
    [1,  0, 0, 0, 0, 1, 0, 0], # RH Med
    [0,  0, 1, 0, 0, 0, 0, 0], # RH Card
    [1,  0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 1, 0, 1, 0, 0], # S2 Esp
    [0,  0, 0, 0, 1, 0, 0, 0], # S Cam Gama
    [0,  0, 0, 0, 0, 0, 0, 0], # S Cortina
    [0,  0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 1, 0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0,  1, 1, 0, 0, 0, 1, 1], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0, 0, 0]  # WC
    ])

rec0_acm = np.array([
    [0,  1, 2, 3, 4, 5, 6, 7], # task
    [0, 10,19,0,13, 0, 0, 0], # duration
    [0,  0, 0, 1, 1, 1, 0, 0], # RH TAS
    [0,  1, 1, 0, 1, 0, 1, 1], # RH Enf
    [0,  0, 0, 0, 1, 0, 0, 0], # RH Tec
    [1,  0, 0, 0, 0, 1, 0, 0], # RH Med
    [0,  0, 1, 0, 0, 0, 0, 0], # RH Card
    [0,  0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 0, 0, 0, 0, 0], # S2 Esp
    [0,  0, 0, 0, 1, 0, 0, 0], # S Cam Gama
    [1,  1, 1, 1, 0, 1, 0, 1], # S Cortina
    [0,  0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0,  0, 0, 0, 0, 0, 1, 0], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 1, 0, 0, 0]  # WC
    ])

#Cintigrafia Miocárdica de perfusão em repouso
rec1_aut = np.array([
    [0 , 1, 2, 3, 4, 5, 6], # task
    [10, 5, 4,59,12,14, 3], # duration
    [0,  0, 0, 1, 1, 1, 0], # RH TAS
    [0,  1, 1, 0, 0, 0, 1], # RH Enf
    [0,  0, 0, 0, 1, 0, 0], # RH Tec
    [1,  0, 0, 0, 0, 1, 0], # RH Med
    [0,  0, 0, 0, 0, 0, 0], # RH Card
    [1,  0, 0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 1, 0, 1, 0], # S2 Esp
    [0,  0, 0, 0, 1, 0, 0], # S Cam Gama
    [0,  0, 0, 0, 0, 0, 0], # S Cortina
    [0,  0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 1
    [0,  1, 1, 0, 0, 0, 1], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0, 0]  # WC
    ])

rec1_acm = np.array([
    [0 , 1, 2, 3, 4, 5, 6], # task
    [0, 4, 13,42,15,8, 0], # duration
    [0,  0, 0, 1, 1, 1, 0], # RH TAS
    [0,  1, 1, 0, 1, 0, 1], # RH Enf
    [0,  0, 0, 0, 1, 0, 0], # RH Tec
    [1,  0, 0, 0, 0, 1, 0], # RH Med
    [0,  0, 0, 0, 0, 0, 0], # RH Card
    [0,  0, 0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 0, 0, 0, 0], # S2 Esp
    [0,  0, 0, 0, 1, 0, 0], # S Cam Gama
    [1,  1, 1, 1, 0, 1, 1], # S Cortina
    [0,  0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 1
    [0,  0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0, 0]  # WC
    ])

#Cintigrafia Miocárdica de perfusão em repouso - estudo de viabilidade
rec2_aut = np.array([
    [0 , 1, 2, 3, 4, 5, 6], # task
    [11, 7, 23,60,13,10, 3], # duration
    [0,  0, 0, 1, 1, 1, 0], # RH TAS
    [0,  1, 1, 0, 0, 0, 1], # RH Enf
    [0,  0, 0, 0, 1, 0, 0], # RH Tec
    [1,  0, 0, 0, 0, 1, 0], # RH Med
    [0,  0, 0, 0, 0, 0, 0], # RH Card
    [1,  0, 0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 1, 0, 1, 0], # S2 Esp
    [0,  0, 0, 0, 1, 0, 0], # S Cam Gama
    [0,  0, 0, 0, 0, 0, 0], # S Cortina
    [0,  0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 1
    [0,  1, 1, 0, 0, 0, 1], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0, 0]  # WC
    ])

rec2_acm = np.array([
    [0 , 1, 2, 3, 4, 5, 6], # task
    [0, 0, 0,0,0,0, 0], # duration
    [0,  0, 0, 1, 1, 1, 0], # RH TAS
    [0,  1, 1, 0, 1, 0, 1], # RH Enf
    [0,  0, 0, 0, 1, 0, 0], # RH Tec
    [1,  0, 0, 0, 0, 1, 0], # RH Med
    [0,  0, 0, 0, 0, 0, 0], # RH Card
    [0,  0, 0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 0, 0, 0, 0], # S2 Esp
    [0,  0, 0, 0, 1, 0, 0], # S Cam Gama
    [1,  1, 1, 1, 0, 1, 1], # S Cortina
    [0,  0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 1
    [0,  0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0, 0]  # WC
    ])

#Cintigrafia Miocárdica de perfusão em esforço/stress farmacológico + repouso
rec3_aut = np.array([
    [0 , 1,  2,  3,  4, 5,  6,  7,  8, 9,10], # task
    [18, 9, 12, 65, 13, 5, 30, 45, 30, 4, 9], # duration
    [0 , 0,  0,  1,  1, 1,  0,  1,  1, 1, 0], # RH TAS
    [0 , 1,  1,  0,  0, 0,  1,  0,  0, 0, 1], # RH Enf
    [0 , 0,  0,  0,  1, 0,  0,  0,  1, 0, 0], # RH Tec
    [1 , 0,  0,  0,  0, 1,  0,  0,  0, 1, 0], # RH Med
    [0 , 0,  1,  0,  0, 0,  0,  0,  0, 0, 0], # RH Card
    [1 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # Gab Med
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S2 Bal
    [0 , 0,  0,  1,  0, 1,  0,  1,  0, 1, 0], # S2 Esp
    [0 , 0,  0,  0,  1, 0,  0,  0,  1, 0, 0], # S Cam Gama
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Cortina
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Esp Criança
    [0 , 0,  1,  0,  0, 0,  0,  0,  0, 0, 0], # S Pol
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Enf 1
    [0 , 1,  1,  0,  0, 0,  1,  0,  0, 0, 1], # S Enf 1 ou S Pol
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Enf 2 
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Enf 2 ou 3
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Tom
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0]  # WC
    ])

rec3_acm = np.array([
    [0 , 1,  2,  3,  4, 5,  6,  7,  8, 9,10], # task
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # duration
    [0 , 0,  0,  1,  1, 1,  0,  1,  1, 1, 0], # RH TAS
    [0 , 1,  1,  0,  0, 0,  1,  0,  0, 0, 1], # RH Enf
    [0 , 0,  0,  0,  1, 0,  0,  0,  1, 0, 0], # RH Tec
    [1 , 0,  0,  0,  0, 1,  0,  0,  0, 1, 0], # RH Med
    [0 , 0,  1,  0,  0, 0,  0,  0,  0, 0, 0], # RH Card
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # Gab Med
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S2 Bal
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S2 Esp
    [0 , 0,  0,  0,  1, 0,  0,  0,  1, 0, 0], # S Cam Gama
    [1 , 1,  1,  1,  0, 1,  1,  1,  0, 1, 1], # S Cortina
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Esp Criança
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Pol
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Enf 1
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Enf 1 ou S Pol
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Enf 2 
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Enf 2 ou 3
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0], # S Tom
    [0 , 0,  0,  0,  0, 0,  0,  0,  0, 0, 0]  # WC
    ])


#Cintigrafia Óssea corpo inteiro
rec4_aut = np.array([
    [0, 1,  2, 3, 4], # task
    [11,7,154,19, 7], # duration
    [0, 0,  0, 1, 1], # RH TAS
    [0, 1,  0, 0, 0], # RH Enf
    [0, 0,  0, 1, 0], # RH Tec
    [1, 0,  0, 0, 1], # RH Med
    [0, 0,  0, 0, 0], # RH Card
    [1, 0,  0, 0, 0], # Gab Med
    [0, 0,  0, 0, 0], # S2 Bal
    [0, 0,  0, 0, 1], # S2 Esp
    [0, 0,  0, 1, 0], # S Cam Gama
    [0, 0,  0, 0, 0], # S Cortina
    [0, 0,  0, 0, 0], # S Esp Criança
    [0, 0,  0, 0, 0], # S Pol
    [0, 0,  0, 0, 0], # S Enf 1
    [0, 1,  0, 0, 0], # S Enf 1 ou S Pol
    [0, 0,  0, 0, 0], # S Enf 2 
    [0, 0,  0, 0, 0], # S Enf 2 ou 3
    [0, 0,  0, 0, 0], # S Tom
    [0, 0,  0, 0, 0]  # WC
    ])

rec4_acm = np.array([
    [0, 1,  2, 3, 4], # task
    [9, 5,156,22, 2], # duration
    [0, 0,  1, 1, 1], # RH TAS
    [0, 1,  0, 1, 0], # RH Enf
    [0, 0,  0, 1, 0], # RH Tec
    [1, 0,  0, 0, 1], # RH Med
    [0, 0,  0, 0, 0], # RH Card
    [0, 0,  0, 0, 0], # Gab Med
    [0, 0,  0, 0, 0], # S2 Bal
    [0, 0,  0, 0, 0], # S2 Esp
    [0, 0,  0, 1, 0], # S Cam Gama
    [1, 1,  1, 0, 1], # S Cortina
    [0, 0,  0, 0, 0], # S Esp Criança
    [0, 0,  0, 0, 0], # S Pol
    [0, 0,  0, 0, 0], # S Enf 1
    [0, 0,  0, 0, 0], # S Enf 1 ou S Pol
    [0, 0,  0, 0, 0], # S Enf 2 
    [0, 0,  0, 0, 0], # S Enf 2 ou 3
    [0, 0,  0, 0, 0], # S Tom
    [0, 0,  0, 0, 0]  # WC
    ])

#Cintigrafia Óssea em três fases
rec5_aut = np.array([
    [0, 1, 2, 3,  4, 5, 6, 7], # task
    [12,4, 8, 4,128,31, 4, 3], # duration
    [0, 0, 1, 1,  0, 1, 1, 0], # RH TAS
    [0, 1, 0, 0,  0, 0, 0, 1], # RH Enf
    [0, 0, 1, 0,  0, 1, 0, 0], # RH Tec
    [1, 0, 0, 1,  0, 0, 1, 0], # RH Med
    [0, 0, 0, 0,  0, 0, 0, 0], # RH Card
    [1, 0, 0, 0,  0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0,  0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 1,  0, 0, 1, 0], # S2 Esp
    [0, 0, 1, 0,  0, 1, 0, 0], # S Cam Gama
    [0, 0, 0, 0,  0, 0, 0, 0], # S Cortina
    [0, 0, 0, 0,  0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0,  0, 0, 0, 0], # S Pol
    [0, 0, 0, 0,  0, 0, 0, 0], # S Enf 1
    [0, 1, 0, 0,  0, 0, 0, 1], # S Enf 1 ou S Pol
    [0, 0, 0, 0,  0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0,  0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0,  0, 0, 0, 0], # S Tom
    [0, 0, 0, 0,  0, 0, 0, 0]  # WC
    ])

rec5_acm = np.array([
    [0, 1, 2, 3,  4, 5, 6, 7], # task
    [0,0, 0, 0,0,0, 0, 0], # duration
    [0, 0, 1, 1,  1, 1, 1, 0], # RH TAS
    [0, 1, 1, 0,  0, 1, 0, 1], # RH Enf
    [0, 0, 1, 0,  0, 1, 0, 0], # RH Tec
    [1, 0, 0, 1,  0, 0, 1, 0], # RH Med
    [0, 0, 0, 0,  0, 0, 0, 0], # RH Card
    [0, 0, 0, 0,  0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0,  0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0,  0, 0, 0, 0], # S2 Esp
    [0, 0, 1, 0,  0, 1, 0, 0], # S Cam Gama
    [1, 1, 0, 1,  1, 0, 1, 1], # S Cortina
    [0, 0, 0, 0,  0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0,  0, 0, 0, 0], # S Pol
    [0, 0, 0, 0,  0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0,  0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0,  0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0,  0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0,  0, 0, 0, 0], # S Tom
    [0, 0, 0, 0,  0, 0, 0, 0]  # WC
    ])

#Cintigrafia para Amiloidose cardíaca
rec6_aut = np.array([
    [0,  1, 2,  3, 4], # task
    [7, 10,230,38, 3], # duration
    [0,  0, 0,  1, 1], # RH TAS
    [0,  1, 0,  0, 0], # RH Enf
    [0,  0, 0,  1, 0], # RH Tec
    [1,  0, 0,  0, 1], # RH Med
    [0,  0, 0,  0, 0], # RH Card
    [1,  0, 0,  0, 0], # Gab Med
    [0,  0, 0,  0, 0], # S2 Bal
    [0,  0, 0,  0, 1], # S2 Esp
    [0,  0, 0,  1, 0], # S Cam Gama
    [0,  0, 0,  0, 0], # S Cortina
    [0,  0, 0,  0, 0], # S Esp Criança
    [0,  0, 0,  0, 0], # S Pol
    [0,  0, 0,  0, 0], # S Enf 1
    [0,  1, 0,  0, 0], # S Enf 1 ou S Pol
    [0,  0, 0,  0, 0], # S Enf 2 
    [0,  0, 0,  0, 0], # S Enf 2 ou 3
    [0,  0, 0,  0, 0], # S Tom
    [0,  0, 0,  0, 0]  # WC
    ])

rec6_acm = np.array([
    [0,  1, 2,  3, 4], # task
    [0, 0,0,0, 0], # duration
    [0,  0, 0,  1, 1], # RH TAS
    [0,  1, 0,  1, 0], # RH Enf
    [0,  0, 0,  1, 0], # RH Tec
    [1,  0, 0,  0, 1], # RH Med
    [0,  0, 0,  0, 0], # RH Card
    [0,  0, 0,  0, 0], # Gab Med
    [0,  0, 0,  0, 0], # S2 Bal
    [0,  0, 0,  0, 0], # S2 Esp
    [0,  0, 0,  1, 0], # S Cam Gama
    [1,  1, 1,  0, 1], # S Cortina
    [0,  0, 0,  0, 0], # S Esp Criança
    [0,  0, 0,  0, 0], # S Pol
    [0,  0, 0,  0, 0], # S Enf 1
    [0,  0, 0,  0, 0], # S Enf 1 ou S Pol
    [0,  0, 0,  0, 0], # S Enf 2 
    [0,  0, 0,  0, 0], # S Enf 2 ou 3
    [0,  0, 0,  0, 0], # S Tom
    [0,  0, 0,  0, 0]  # WC
    ])

#Cintigrafia Pulmonar de ventilação/inalação + perfusão
rec7_aut = np.array([
    [0,  1, 2, 3, 4, 5], # task
    [13, 5,24,11,13, 4], # duration
    [0,  0, 1, 1, 1, 1], # RH TAS
    [0,  0, 0, 1, 0, 0], # RH Enf
    [0,  1, 1, 1, 1, 0], # RH Tec
    [1,  0, 0, 0, 0, 1], # RH Med
    [0,  0, 0, 0, 0, 0], # RH Card
    [1,  0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 0, 0, 1], # S2 Esp
    [0,  0, 1, 1, 1, 0], # S Cam Gama
    [0,  0, 0, 0, 0, 0], # S Cortina
    [0,  0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0], # S Enf 1
    [0,  0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0,  1, 1, 1, 1, 0], # S Enf 2 
    [0,  3, 3, 3, 3, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0]  # WC
    ])

rec7_acm = np.array([
    [0,  1, 2, 3, 4, 5], # task
    [13, 5,12,5 ,16, 6], # duration
    [0,  0, 1, 1, 1, 1], # RH TAS
    [0,  0, 0, 1, 0, 0], # RH Enf
    [0,  1, 1, 1, 1, 0], # RH Tec
    [1,  0, 0, 0, 0, 1], # RH Med
    [0,  0, 0, 0, 0, 0], # RH Card
    [0,  0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 0, 0, 0], # S2 Esp
    [0,  0, 1, 1, 1, 0], # S Cam Gama
    [1,  0, 0, 0, 0, 1], # S Cortina
    [0,  0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0], # S Enf 1
    [0,  0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0,  1, 1, 1, 1, 0], # S Enf 2 
    [0,  3, 3, 3, 3, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0]  # WC
    ])

#Cintigrafia Renal com 99mTc-DMSA
rec8_cri = np.array([
    [0,  1,  2, 3, 4], # task
    [13,11,199,37, 5], # duration
    [0,  1,  0, 1, 1], # RH TAS
    [0,  2,  0, 0, 0], # RH Enf
    [0,  0,  0, 1, 0], # RH Tec
    [1,  0,  0, 0, 1], # RH Med
    [0,  0,  0, 0, 0], # RH Card
    [1,  0,  0, 0, 0], # Gab Med
    [0,  0,  0, 0, 0], # S2 Bal
    [0,  0,  0, 0, 1], # S2 Esp
    [0,  0,  0, 1, 0], # S Cam Gama
    [0,  0,  0, 0, 0], # S Cortina
    [0,  0,  0, 0, 0], # S Esp Criança
    [0,  0,  0, 0, 0], # S Pol
    [0,  0,  0, 0, 0], # S Enf 1
    [0,  1,  0, 0, 0], # S Enf 1 ou S Pol
    [0,  0,  0, 0, 0], # S Enf 2 
    [0,  0,  0, 0, 0], # S Enf 2 ou 3
    [0,  0,  0, 0, 0], # S Tom
    [0,  0,  0, 0, 0]  # WC
    ])

#Cintigrafia Tiroideia
rec9_aut = np.array([
    [0, 1, 2, 3, 4], # task
    [9, 7,16,12, 4], # duration
    [0, 0, 1, 1, 1], # RH TAS
    [0, 1, 0, 0, 0], # RH Enf
    [0, 0, 0, 1, 0], # RH Tec
    [1, 0, 0, 0, 1], # RH Med
    [0, 0, 0, 0, 0], # RH Card
    [1, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 1, 0, 1], # S2 Esp
    [0, 0, 0, 1, 0], # S Cam Gama
    [0, 0, 0, 0, 0], # S Cortina
    [0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0], # S Pol
    [0, 0, 0, 0, 0], # S Enf 1
    [0, 1, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0], # S Tom
    [0, 0, 0, 0, 0]  # WC
    ])

rec9_acm = np.array([
    [0, 1, 2, 3, 4], # task
    [0, 0,0,0, 0], # duration
    [0, 0, 1, 1, 1], # RH TAS
    [0, 1, 0, 1, 0], # RH Enf
    [0, 0, 0, 1, 0], # RH Tec
    [1, 0, 0, 0, 1], # RH Med
    [0, 0, 0, 0, 0], # RH Card
    [0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0], # S2 Esp
    [0, 0, 0, 1, 0], # S Cam Gama
    [1, 1, 1, 0, 1], # S Cortina
    [0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0], # S Pol
    [0, 0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0], # S Tom
    [0, 0, 0, 0, 0]  # WC
    ])

#Cintigrafia das Glândulas salivares
"""
ESTIMATIVAS, SEM DADOS NENHUNS
"""
rec10_aut = np.array([
    [0,  1, 2, 3, 4], # task
    [15,15,60,10, 5], # duration
    [0,  0, 1, 1, 0], # RH TAS
    [0,  1, 1, 0, 1], # RH Enf
    [0,  0, 1, 0, 0], # RH Tec
    [1,  0, 0, 1, 0], # RH Med
    [0,  0, 0, 0, 0], # RH Card
    [1,  0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 1, 0], # S2 Esp
    [0,  0, 1, 0, 0], # S Cam Gama
    [0,  0, 0, 0, 0], # S Cortina
    [0,  0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0], # S Enf 1
    [0,  1, 0, 0, 1], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0]  # WC
    ])

rec10_acm = np.array([
    [0,  1, 2, 3, 4], # task
    [15,15,60,10, 5], # duration
    [0,  0, 1, 1, 0], # RH TAS
    [0,  1, 1, 0, 1], # RH Enf
    [0,  0, 1, 0, 0], # RH Tec
    [1,  0, 0, 1, 0], # RH Med
    [0,  0, 0, 0, 0], # RH Card
    [0,  0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 0, 0], # S2 Esp
    [0,  0, 1, 0, 0], # S Cam Gama
    [1,  1, 0, 1, 1], # S Cortina
    [0,  0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0], # S Enf 1
    [0,  0, 0, 0, 0], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0]  # WC
    ])

#Cintigrafia das Paratiroideias
rec11_aut = np.array([
    [0, 1, 2, 3,  4, 5,6], # task
    [10,6, 0,30,180,19,4], # duration
    [0, 0, 1, 1, 0, 1, 1], # RH TAS
    [0, 1, 0, 0, 0, 0, 0], # RH Enf
    [0, 0, 0, 1, 0, 1, 0], # RH Tec
    [1, 0, 0, 0, 0, 0, 1], # RH Med
    [0, 0, 0, 0, 0, 0, 0], # RH Card
    [1, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 1, 0, 0, 0, 1], # S2 Esp
    [0, 0, 0, 1, 0, 1, 0], # S Cam Gama
    [0, 0, 0, 0, 0, 0, 0], # S Cortina
    [0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 1, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0]  # WC
    ])

rec11_acm = np.array([
    [0, 1, 2, 3,  4, 5,6], # task
    [0,0, 0,0,0,0,0], # duration
    [0, 0, 1, 1, 0, 1, 1], # RH TAS
    [0, 1, 0, 1, 0, 1, 0], # RH Enf
    [0, 0, 0, 1, 0, 1, 0], # RH Tec
    [1, 0, 0, 0, 0, 0, 1], # RH Med
    [0, 0, 0, 0, 0, 0, 0], # RH Card
    [0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0, 0, 0], # S2 Esp
    [0, 0, 0, 1, 0, 1, 0], # S Cam Gama
    [1, 1, 1, 0, 1, 0, 1], # S Cortina
    [0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0]  # WC
    ])

#Linfocintigrafia para Detecção de gânglio sentinela
rec12_aut = np.array([
    [0,1, 2, 3, 4, 5, 6], # task
    [7,7,60, 10,30,14, 4], # duration
    [0,0, 1, 0, 1, 1, 1], # RH TAS
    [0,1, 0, 1, 0, 0, 0], # RH Enf
    [0,0, 0, 1, 0, 1, 0], # RH Tec
    [1,0, 0, 0, 0, 0, 1], # RH Med
    [0,0, 0, 0, 0, 0, 0], # RH Card
    [1,0, 0, 0, 0, 0, 0], # Gab Med
    [0,0, 0, 0, 0, 0, 0], # S2 Bal
    [0,0, 1, 0, 1, 0, 1], # S2 Esp
    [0,0, 0, 0, 0, 1, 0], # S Cam Gama
    [0,0, 0, 0, 0, 0, 0], # S Cortina
    [0,0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,0, 0, 1, 0, 0, 0], # S Pol
    [0,0, 0, 0, 0, 0, 0], # S Enf 1
    [0,1, 0, 1, 0, 0, 0], # S Enf 1 ou S Pol
    [0,0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,0, 0, 0, 0, 0, 0], # S Tom
    [0,0, 0, 0, 0, 0, 0]  # WC
    ])

rec12_acm = np.array([
    [0,1, 2, 3, 4, 5, 6], # task
    [0,0,0, 0,0,0, 0], # duration
    [0,0, 1, 0, 1, 1, 1], # RH TAS
    [0,1, 0, 1, 0, 1, 0], # RH Enf
    [0,0, 0, 1, 0, 1, 0], # RH Tec
    [1,0, 0, 0, 0, 0, 1], # RH Med
    [0,0, 0, 0, 0, 0, 0], # RH Card
    [0,0, 0, 0, 0, 0, 0], # Gab Med
    [0,0, 0, 0, 0, 0, 0], # S2 Bal
    [0,0, 0, 0, 0, 0, 0], # S2 Esp
    [0,0, 0, 0, 0, 1, 0], # S Cam Gama
    [1,1, 1, 1, 1, 0, 1], # S Cortina
    [0,0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,0, 0, 0, 0, 0, 0], # S Pol
    [0,0, 0, 0, 0, 0, 0], # S Enf 1
    [0,0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0,0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,0, 0, 0, 0, 0, 0], # S Tom
    [0,0, 0, 0, 0, 0, 0]  # WC
    ])

"""
#Renograma Simples ou F0
rec13 = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8], # task
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # duration
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH TAS
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Enf
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Tec
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Med
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Card
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S2 Esp
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Cam Gama
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Cortina
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 3
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0, 0, 0]  # WC
    ])
"""

#Renograma F+20
rec14_aut = np.array([
    [0,  1, 2, 3, 4, 5, 6, 7], # task
    [15,10,28,30, 5, 5, 0, 0], # duration
    [0,  0, 1, 1, 1, 0, 1, 0], # RH TAS
    [0,  1, 1, 0, 0, 0, 0, 1], # RH Enf
    [0,  0, 1, 0, 1, 0, 1, 0], # RH Tec
    [1,  0, 0, 0, 0, 1, 0, 0], # RH Med
    [0,  0, 0, 0, 0, 0, 0, 0], # RH Card
    [1,  0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 1, 0, 0, 0, 0], # S2 Esp
    [0,  0, 1, 0, 1, 1, 1, 0], # S Cam Gama
    [0,  0, 0, 0, 0, 0, 0, 0], # S Cortina
    [0,  0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0,  1, 0, 0, 0, 0, 0, 1], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0, 0, 0, 0], # S Enf 2
    [0,  0, 0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0, 0, 0]  # WC
    ])

#Cistocintigrafia Indirecta
rec15_aut = np.array([
    [0, 1, 2, 3], # task
    [0,24,39,11], # duration
    [0, 1, 1, 1], # RH TAS
    [0, 0, 1, 0], # RH Enf
    [0, 0, 1, 0], # RH Tec
    [0, 0, 0, 1], # RH Med
    [0, 0, 0, 0], # RH Card
    [0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0], # S2 Bal
    [0, 1, 0, 1], # S2 Esp
    [0, 0, 1, 0], # S Cam Gama
    [0, 0, 0, 0], # S Cortina
    [0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0], # S Pol
    [0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0], # S Tom
    [0, 0, 0, 0]  # WC
    ])

rec15_acm = np.array([
    [0, 1, 2, 3], # task
    [0, 0,39,10], # duration
    [0, 1, 1, 1], # RH TAS
    [0, 0, 1, 0], # RH Enf
    [0, 0, 1, 0], # RH Tec
    [0, 0, 0, 1], # RH Med
    [0, 0, 0, 0], # RH Card
    [0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0], # S2 Esp
    [0, 0, 1, 0], # S Cam Gama
    [0, 1, 0, 1], # S Cortina
    [0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0], # S Pol
    [0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0], # S Tom
    [0, 0, 0, 0]  # WC
    ])

#Tomografia cerebral com 123I-Ioflupano
rec16_aut = np.array([
    [0,  1, 2, 3, 4, 5, 6], # task
    [12,20,60,20,180,48, 4], # duration
    [0,  0, 0, 0, 0, 1, 1], # RH TAS
    [0,  1, 0, 1, 0, 0, 0], # RH Enf
    [0,  0, 0, 0, 0, 1, 0], # RH Tec
    [1,  0, 0, 0, 0, 0, 1], # RH Med
    [0,  0, 0, 0, 0, 0, 0], # RH Card
    [1,  0, 0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 0, 0, 0, 1], # S2 Esp
    [0,  0, 0, 0, 0, 1, 0], # S Cam Gama
    [0,  0, 0, 0, 0, 0, 0], # S Cortina
    [0,  0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 1
    [0,  1, 0, 1, 0, 0, 0], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0, 0]  # WC
    ])

rec16_acm = np.array([
    [0,  1, 2, 3, 4, 5, 6], # task
    [0,0,0,0,0,0,0], # duration
    [0,  0, 0, 0, 0, 1, 1], # RH TAS
    [0,  1, 0, 1, 0, 1, 0], # RH Enf
    [0,  0, 0, 0, 0, 1, 0], # RH Tec
    [1,  0, 0, 0, 0, 0, 1], # RH Med
    [0,  0, 0, 0, 0, 0, 0], # RH Card
    [0,  0, 0, 0, 0, 0, 0], # Gab Med
    [0,  0, 0, 0, 0, 0, 0], # S2 Bal
    [0,  0, 0, 0, 0, 0, 0], # S2 Esp
    [0,  0, 0, 0, 0, 1, 0], # S Cam Gama
    [1,  1, 1, 1, 1, 0, 1], # S Cortina
    [0,  0, 0, 0, 0, 0, 0], # S Esp Criança
    [0,  0, 0, 0, 0, 0, 0], # S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 1
    [0,  0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 
    [0,  0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0,  0, 0, 0, 0, 0, 0], # S Tom
    [0,  0, 0, 0, 0, 0, 0]  # WC
    ])

#PET - Estudo corpo inteiro com FDG											
rec17_aut = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7], # task
    [11,8,30, 5,54,20, 5, 7], # duration
    [0, 0, 0, 0, 0, 1, 1, 0], # RH TAS
    [0, 1, 0, 1, 0, 1, 0, 1], # RH Enf
    [0, 0, 0, 0, 0, 1, 0, 0], # RH Tec
    [1, 0, 0, 0, 0, 0, 1, 0], # RH Med
    [0, 0, 0, 0, 0, 0, 0, 0], # RH Card
    [1, 0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0, 0, 1, 0], # S2 Esp
    [0, 0, 0, 0, 0, 0, 0, 0], # S Cam Gama
    [0, 0, 0, 0, 0, 0, 0, 0], # S Cortina
    [0, 0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 1, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 1, 0, 0, 0, 0, 0, 1], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 1, 1, 1, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 1, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0, 0]  # WC
    ])

rec17_acm = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7], # task
    [14,9,33, 0, 0,23, 3, 0], # duration
    [0, 0, 0, 0, 0, 1, 1, 0], # RH TAS
    [0, 1, 0, 1, 0, 1, 0, 1], # RH Enf
    [0, 0, 0, 0, 0, 1, 0, 0], # RH Tec
    [1, 0, 0, 0, 0, 0, 1, 0], # RH Med
    [0, 0, 0, 0, 0, 0, 0, 0], # RH Card
    [0, 0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0, 0, 0, 0], # S2 Esp
    [0, 0, 0, 0, 0, 0, 0, 0], # S Cam Gama
    [1, 1, 0, 0, 0, 0, 1, 1], # S Cortina
    [0, 0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 1, 1, 1, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 1, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0, 0]  # WC
    ])

#PET - Com 68 Ga-Péptidos										
rec18_aut = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7], # task
    [15,8,63, 0,0,40, 4, 3], # duration
    [0, 0, 0, 0, 0, 1, 1, 0], # RH TAS
    [0, 1, 0, 1, 0, 1, 0, 1], # RH Enf
    [0, 0, 0, 0, 0, 1, 0, 0], # RH Tec
    [1, 0, 0, 0, 0, 0, 1, 0], # RH Med
    [0, 0, 0, 0, 0, 0, 0, 0], # RH Card
    [1, 0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0, 0, 1, 0], # S2 Esp
    [0, 0, 0, 0, 0, 0, 0, 0], # S Cam Gama
    [0, 0, 0, 0, 0, 0, 0, 0], # S Cortina
    [0, 0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 1, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 1, 0, 0, 0, 0, 0, 1], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 1, 1, 1, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 1, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0, 0]  # WC
    ])

rec18_acm = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7], # task
    [0,0,0, 0, 0,0, 0, 0], # duration
    [0, 0, 0, 0, 0, 1, 1, 0], # RH TAS
    [0, 1, 0, 1, 0, 1, 0, 1], # RH Enf
    [0, 0, 0, 0, 0, 1, 0, 0], # RH Tec
    [1, 0, 0, 0, 0, 0, 1, 0], # RH Med
    [0, 0, 0, 0, 0, 0, 0, 0], # RH Card
    [0, 0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0, 0, 0, 0], # S2 Esp
    [0, 0, 0, 0, 0, 0, 0, 0], # S Cam Gama
    [1, 1, 0, 0, 0, 0, 1, 1], # S Cortina
    [0, 0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 1, 1, 1, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 1, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0, 0]  # WC
    ])

#PET - Com 68Ga-PSMA										
rec19_aut = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7], # task
    [13,5,38,60,0,24, 7, 4], # duration
    [0, 0, 0, 0, 0, 1, 1, 0], # RH TAS
    [0, 1, 0, 1, 0, 1, 0, 1], # RH Enf
    [0, 0, 0, 0, 0, 1, 0, 0], # RH Tec
    [1, 0, 0, 0, 0, 0, 1, 0], # RH Med
    [0, 0, 0, 0, 0, 0, 0, 0], # RH Card
    [1, 0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0, 0, 1, 0], # S2 Esp
    [0, 0, 0, 0, 0, 0, 0, 0], # S Cam Gama
    [0, 0, 0, 0, 0, 0, 0, 0], # S Cortina
    [0, 0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 1, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 1, 0, 0, 0, 0, 0, 1], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 1, 1, 1, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 1, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0, 0]  # WC
    ])

rec19_acm = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7], # task
    [18,5,0, 0, 35,0, 2, 0], # duration
    [0, 0, 0, 0, 0, 1, 1, 0], # RH TAS
    [0, 1, 0, 1, 0, 1, 0, 1], # RH Enf
    [0, 0, 0, 0, 0, 1, 0, 0], # RH Tec
    [1, 0, 0, 0, 0, 0, 1, 0], # RH Med
    [0, 0, 0, 0, 0, 0, 0, 0], # RH Card
    [0, 0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0, 0, 0, 0], # S2 Esp
    [0, 0, 0, 0, 0, 0, 0, 0], # S Cam Gama
    [1, 1, 0, 0, 0, 0, 1, 1], # S Cortina
    [0, 0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 1, 1, 1, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 1, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0, 0]  # WC
    ])


"""
rec4 = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8], # task
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # duration
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH TAS
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Enf
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Tec
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Med
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # RH Card
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # Gab Med
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S2 Bal
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S2 Esp
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Cam Gama
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Cortina
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Esp Criança
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Pol
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 1 ou S Pol
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Enf 2 ou 3
    [0, 0, 0, 0, 0, 0, 0, 0, 0], # S Tom
    [0, 0, 0, 0, 0, 0, 0, 0, 0]  # WC
    ])
"""

"""
rec0  -Cintigrafia Miocárdica de perfusão em esforço/stress farmacológico
rec1  -Cintigrafia Miocárdica de perfusão em repouso
rec2  -Cintigrafia Miocárdica de perfusão em repouso - estudo de viabilidade
rec3  -Cintigrafia Miocárdica de perfusão em esforço/stress farmacológico + repouso
rec4  -Cintigrafia Óssea corpo inteiro
rec5  -Cintigrafia Óssea em três fases
rec6  -Cintigrafia para Amiloidose cardíaca
rec7  -Cintigrafia Pulmonar de ventilação/inalação + perfusão
rec8  -Cintigrafia Renal com 99mTc-DMSA
rec9  -Cintigrafia Tiroideia
rec10 -Cintigrafia das Glândulas salivares
rec11 -Cintigrafia das Paratiroideias
rec12 -Linfocintigrafia para Detecção de gânglio sentinela
rec13 -Renograma Simples ou F0
rec14 -Renograma F+20
rec15 -Cistocintigrafia Indirecta
rec16 -Tomografia cerebral com 123I-Ioflupano
rec17 -PET - Estudo corpo inteiro com FDG		
rec18 -PET - Com 68 Ga-Péptidos		
rec19 -PET - Com 68Ga-PSMA
"""

rec = [rec9_aut, rec7_aut, rec1_aut, rec10_aut, rec17_aut] #Segunda [3, 5, 10, 1, 10] pro_i = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
#rec = [rec9_aut, rec4_aut, rec6_aut, rec12_aut, rec7_aut, rec17_aut] #Terça [2, 12, 1, 1, 2, 10] pro_i = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
#rec = [rec9_aut, rec12_aut, rec11_aut, rec16_aut, rec7_aut, rec17_aut] #Quarta [2, 2, 6, 1, 3, 8] pro_i = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5])
#rec = [rec12_aut, rec0_aut, rec16_aut, rec1_aut, rec17_aut] #Quinta [6, 12, 1, 2, 10] pro_i = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
#rec = [rec0_aut, rec1_aut, rec4_aut, rec6_aut, rec7_aut, rec9_aut, rec10_aut, rec11_aut, rec12_aut, rec16_aut, rec17_aut] #[12, 12, 12, 1, 10, 7, 1, 6, 9, 2, 38]
#rec = [rec0_aut, rec1_aut, rec2_aut, rec3_aut, rec4_aut, rec5_aut, rec6_aut, rec7_aut, rec8_cri, rec9_aut, rec11_aut, rec12_aut, rec14_aut, rec15_aut, rec16_aut, rec17_aut, rec18_aut, rec19_aut]


exams_to_assign(rec)


















