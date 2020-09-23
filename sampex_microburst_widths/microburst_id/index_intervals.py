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
    MOD:     2017-02-15
    """
    conv = np.convolve([1, -1], x, mode = 'valid') - 1
    consecutiveFlag = np.where(conv != 0)[0] + 1
    startInd = np.insert(consecutiveFlag, 0, 0)
    endInd = np.insert(consecutiveFlag, len(consecutiveFlag), len(x)-1)
    return startInd, endInd