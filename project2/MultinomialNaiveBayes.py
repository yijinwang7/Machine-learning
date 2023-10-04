import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline

np.random.seed(1234)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import csc_matrix, csr_matrix


class MultinomialNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X_train, y_train):
        m, n = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

        # init: Prior & Likelihood
        self._priors = np.zeros(n_classes)
        self._likelihoods = np.zeros((n_classes, n))

        # Get Prior and Likelihood
        for idx, c in enumerate(self._classes):
            X_train_c = X_train[c == y_train]
            self._priors[idx] = X_train_c.shape[0] / m
            self._likelihoods[idx, :] = ((X_train_c.sum(axis=0)) + self.alpha) / (
                np.sum(X_train_c.sum(axis=0) + self.alpha))

    def predict(self, X_test):
        prediction = []
        prior_c = np.log(self._priors)
        log_like = csr_matrix(np.log(self._likelihoods)).transpose()
        likelihoods_c = X_test * log_like
        for i in range(X_test.shape[0]):
            temp = likelihoods_c.getrow(i)
            posteriors = temp + prior_c

            prediction.append(self._classes[np.argmax(posteriors)])
        return prediction


    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test) / len(y_test)



