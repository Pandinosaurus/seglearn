'''
===================
Predict Time Series
===================

In this example, we use the pipeline to conduct quasi sequence to sequence predictions

'''
# Author: David Burns
# License: BSD


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from seglearn.pipe import Pype
from seglearn.split import temporal_split
from seglearn.transform import FeatureRep, SegmentXY, last
from seglearn.base import TS_Data

# for a single time series, we need to make it a list
X = [np.arange(10000) / 100.]
y = [np.sin(X[0]) * X[0] * 3 + X[0] * X[0]]
t = [np.arange(len(y[0]))]

X = TS_Data(X, timestamps=t)

# split the data along the time axis (our only option since we have only 1 time series)
X_train, X_test, y_train, y_test = temporal_split(X, y)

# SegmentXY segments both X and y (as the name implies)
# setting y_func = last, selects the last value from each y segment as the target
# other options include transform.middle, or you can make your own function
# see the API documentation for further details

pipe = Pype([('seg', SegmentXY(width=200, overlap=0.5, y_func=last)),
             ('features', FeatureRep()),
             ('lin', LinearRegression())])

# fit and score
pipe.fit(X_train, y_train)

# Pype.predict_series() provides timestamps in addition to the predictions themselves
tp, yp = pipe.predict_series(X_test)

# plot the sequence prediction
plt.plot(X_train.timestamps[0], y_train[0], '.', label="train")
plt.plot(X_test.timestamps[0], y_test[0], '.', label="test")
plt.plot(tp[0], yp[0], '.', label="predict")
plt.xlabel("Time")
plt.ylabel("Target")
plt.legend()
plt.show()

