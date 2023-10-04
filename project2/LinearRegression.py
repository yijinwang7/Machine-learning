
import numpy as np
import matplotlib.pyplot as plt

import warnings

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        pass

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]  # add a dimension for the features
        N = x.shape[0]
        if self.add_bias:
            temp = []
            for i in range (x.shape[0]):
                temp.append([1])
            x = sparse.hstack([x, csr_matrix(temp)]) # add bias by adding a constant feature of value 1
        # alternatively: self.w = np.linalg.inv(x.T @ x)@x.T@y
        #self.w = np.linalg.lstsq(x, y)[0]  # return w for the least square difference
        self.w = sparse.linalg.lsqr(x, y)[0]
        return self

    def predict(self, x):
        if self.add_bias:
            temp = []
            for i in range(x.shape[0]):
                temp.append([1])
            x = sparse.hstack([x, csr_matrix(temp)])
        temp = self.w
        yh = x * temp  # predict the y values
        return np.round(yh.data,0)



