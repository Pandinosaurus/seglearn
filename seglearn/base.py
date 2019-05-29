'''
This module has some base classes for time series data
'''

# Author: David Burns
# License: BSD

import numpy as np

__all__ = ['TS_Data']


class TS_Data(object):
    '''
    Iterable/indexable class for time series data with context data
    Numpy arrays are sufficient time series data alone is needed

    Parameters
    ----------
    ts_data : array-like, shape (N, )
        time series data
    context_data : array-like (N, )
        contextual data

    '''

    def __init__(self, ts_data, context_data=None, timestamps=None):
        N = len(ts_data)
        self.ts_data = np.atleast_1d(ts_data)
        self.context_data = np.atleast_1d(context_data) if context_data is not None else None
        self.timestamps = np.atleast_1d(timestamps) if timestamps is not None \
            else np.array([np.arange(len(ts_data[i])) for i in np.arange(N)])
        self.index = 0
        self.N = N
        self.shape = [N]  # need for safe_indexing with sklearn

    @classmethod
    def from_df(cls, df):
        ts_data = np.array(df['ts_data'])
        timestamp = np.array(df['timestamps']) if 'timestamps' in df else None
        context_data = df.drop(columns=['ts_data', 'timestamps'], errors='ignore')
        context_data = np.array(context_data) if not context_data.empty else None

        return cls(ts_data, context_data, timestamp)

    def concat(self):
        pass

    def __iter__(self):
        return self

    def __getitem__(self, indices):
        ts_data = self.ts_data[indices]
        context_data = self.context_data[indices] if self.context_data is not None else None
        timestamps = self.timestamps[indices]

        return TS_Data(ts_data, context_data, timestamps)
        # return ts_data

    def __next__(self):
        if self.index == self.N:
            raise StopIteration
        self.index = self.index + 1
        return TS_Data(self.__getitem__(self.index))
        # return TS_Data(self.ts_data[self.index], self.context_data[self.index])

    def __len__(self):
        return self.N