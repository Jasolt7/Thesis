# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:20:00 2025

@author: JAPGc
"""

from functools import lru_cache

def distance_func(p1, p2, max_rows):
    x1, y1 = p1
    x2, y2 = p2
    h_dist = abs(x1 - x2)
    if h_dist != 0:
        v_dist = min((2*max_rows) - y1 - y2, (y1 + y2))
    else:
        v_dist = abs(y1 - y2)
    return h_dist + v_dist

@lru_cache(maxsize=None)
def get_distance(p1, p2, max_rows):
    return distance_func(p1, p2, max_rows)