# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:04:52 2025

@author: JAPGc
"""

import numpy as np

# Load the .npy file
data = np.load("rec_lpk.npy")  # This should be a NumPy array of float32 (or whatever type you're using)

# Optional: ensure the dtype is float32 to match your C `float`
data = data.astype(np.float32)

# Save as raw binary
data.tofile("rec_lpk.bin")
