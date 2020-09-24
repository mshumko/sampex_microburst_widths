import numpy as np

def get_index_intervals(x):
    """
    NAME:    get_index_intervals(x)
    USE:     Locates consecutive numbers (indices) in an array x.
             for ith inerval, that interval in x is given by
             x[start_index[i]:end_index[i]]. Technically end_index[i]
             is one greater because Python's indexing excludes the last
             index, i.e. x[0:5] returns x_0, x_1,... x_4 and NOT x_5.
    INPUT:   An sorted array of integers.
    RETURNS: start_index and end_index arrays of consecutive numbers.
    AUTHOR:  Mykhaylo Shumko
    MOD:     2020-09-23
    """
    dx = (x[1:] - x[:-1])
    interval_beaks = np.where(dx-1)[0] + 1
    start_index = np.insert(interval_beaks, 0, 0)
    end_index = np.insert(interval_beaks, len(interval_beaks), len(x))
    return start_index, end_index