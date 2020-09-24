# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:12:10 2017

@author: mike
"""
import numpy as np

def get_index_intervals(x):
    """
    NAME:    get_index_intervals(x)
    USE:     Locates consecutive numbers (indices) in an array x.
    INPUT:   An sorted array of numbers.
    RETURNS: startInd and endInd arrays of consecutive numbers.
    AUTHOR:  Mykhaylo Shumko
    MOD:     2020-09-23
    """
    dx = (x[1:] - x[:-1])
    interval_beaks = np.where(dx-1)[0] + 1
    start_index = np.insert(interval_beaks, 0, 0)
    end_index = np.insert(interval_beaks, len(interval_beaks), len(x))
    return start_index, end_index