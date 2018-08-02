'''
This module has utilities for time series data input checking
'''
# Author: David Burns
# License: BSD

from seglearn.base import TS_Data

import numpy as np

def get_ts_data_parts(X):
    '''
    Separates time series data object into time series variables and contextual variables

    Parameters
    ----------
    X : array-like, shape [n_series, ...]
       Time series data and (optionally) contextual data

    Returns
    -------
    Xt : array-like, shape [n_series, ]
        Time series data
    Xs : array-like, shape [n_series, n_contextd = np.colum _variables]
        contextual variables

    '''
    # todo: remove np array
    if type(X) is TS_Data:
        return X.ts_data, X.context_data
    else:
        return np.array(X), None




def check_ts_data(X, y = None):
    '''
    Checks time series data is good. If not raises assertion error.

    Parameters
    ----------
    X : array-like, shape [n_series, ...]
       Time series data and (optionally) contextual data

    '''
    Ns = len(X)
    Ntx = np.array([len(X[i]) for i in np.arange(Ns)])

    if y is not None:
        assert Ns == len(y)
        Nty = np.array([len(np.atleast_1d(y[i])) for i in np.arange(Ns)])


        if np.count_nonzero(Nty == 1) == Ns:
            return
        elif np.count_nonzero(Nty == Ntx) == Ns:
            return
        else:
            raise TypeError



def ts_stats(Xt, y, fs = 1.0, class_labels = None):
    '''
    Generates some helpful statistics about the data X

    Parameters
    ----------
    X : array-like, shape [n_series, ...]
       Time series data and (optionally) contextual data
    y : array-like, shape [n_series]
        target data
    fs : float
        sampling frequency
    class_labels : list of strings, default None
        List of target class names


    Returns
    -------
    results : dict
        | Dictionary of relevant statistics for the time series data
        | results['total'] has stats for the whole data set
        | results['by_class'] has stats segragated by target class

    '''
    check_ts_data(Xt)
    Xt, Xs = get_ts_data_parts(Xt)

    if Xs is not None:
        S = len(np.atleast_1d(Xs[0]))
    else:
        S = 0

    C = np.max(y) + 1 # number of classes
    if class_labels is None:
        class_labels = np.arange(C)

    N = len(Xt)
    if Xt[0].ndim > 1:
        D = Xt[0].shape[1]
    else:
        D = 1

    Ti = np.array([Xt[i].shape[0] for i in range(N)], dtype=np.float64) / fs
    ic = np.array([y == i for i in range(C)])
    Tic = [Ti[ic[i]] for i in range(C)]

    T = np.sum(Ti)

    total = {"n_series": N, "n_classes": C, "n_TS_vars": D, "n_context_vars": S, "Total_Time": T,
              "Series_Time_Mean": np.mean(Ti),
              "Series_Time_Std": np.std(Ti),
              "Series_Time_Range": (np.min(Ti), np.max(Ti))}

    by_class = {"Class_labels": class_labels,
                "n_series": np.array([len(Tic[i]) for i in range(C)]),
                "Total_Time": np.array([np.sum(Tic[i]) for i in range(C)]),
                "Series_Time_Mean": np.array([np.mean(Tic[i]) for i in range(C)]),
                "Series_Time_Std": np.array([np.std(Tic[i]) for i in range(C)]),
                "Series_Time_Min": np.array([np.min(Tic[i]) for i in range(C)]),
                "Series_Time_Max": np.array([np.max(Tic[i]) for i in range(C)])}

    results = {'total': total,
               'by_class': by_class}

    return results















