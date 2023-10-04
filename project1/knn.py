#!/usr/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import csv
from math import sqrt
import main


# static method
def euclidean_distance(point_1, point_2):
    distance = 0.
    x = point_1.flatten()
    y = point_2.flatten()

    for i in range(min(len(x), len(y))):
        distance += ((float)(x[i]) - (float)(y[i])) ** 2
    return sqrt(distance)


def manhattan_distance(point_1, point_2):
    distance = 0.
    x = point_1.flatten()
    y = point_2.flatten()

    for i in range(min(len(x), len(y))):
        distance += abs((float)(x[i]) - (float)(y[i]))
    return distance

def minkowski_distance(point_1, point_2,p):
    distance = 0.
    x = point_1.flatten()
    y = point_2.flatten()

    for i in range(min(len(x), len(y))):
        distance += ((float)(x[i]) - (float)(y[i]))**p

    d =  abs(distance) ** (1.0 / p)
    return d





def takeFirst(elem):
    return elem[0]


class KNN(object):

    # constructor
    def __init__(self, k):
        self.K = k  # hyper parameter K
        self.training_data = None

    # training of KNN only record the data
    # make the label as value
    def fit(self, training_data):
        self.training_data = training_data

    def predict(self, targets,key="euclidean",p=3):
        predictions = list()
        for target in targets:
            # calculate the distances of the target and every instances in the training set
            distance = list()
            for x in self.training_data:
                # a 2-tuple, first element is the distance, second element is the instance
                d = 0
                if key == "euclidean":
                    d = euclidean_distance(x[:-1], target)
                elif key == "manhattan":
                    d = manhattan_distance(x[:-1], target)
                elif key == "minkowski":
                    d =  minkowski_distance(x[:-1], target,p)

                distance.append((d, x))
                #distance.append((minkowski_distance(x[:-1], target,3), x))
            # sort the list by distance
            distance.sort(key=takeFirst)
            knn = list()
            # get the first k elements
            for i in range(self.K):
                knn.append(distance[i][1])
            # predict the label with max frequency
            output_values = [row[-1] for row in knn]
            target_prediction = max(set(output_values), key=output_values.count)
            predictions.append(target_prediction)
        return predictions


def run_knn(train_set, test_set, k,key ="euclidean",p = 3):
    #### run KNN:
    knn = KNN(k)
    # Now we could do the operations on our data
    knn.fit(train_set)  # train the data
    y_head = knn.predict(test_set,key,p)  # use the test data set to get the prediction y^
    return y_head


def plot_k(train_set, test_set, y):
    k_range = np.arange(1, 50)
    accuracies = list()
    for k in k_range:
        predictions = run_knn(train_set, test_set, k)
        accuracies.append(main.evaluate_acc(y, predictions))

    plt.plot(k_range, accuracies)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different K value in KNN ')
    plt.show()


def plot_distance_function(train_set,test_set,y,):
    k_range = np.arange(1, 50)
    accuracies_eul = list()
    accuracies_man = list()
    accuracies_min = list()
    for k in k_range:
        predictions = run_knn(train_set, test_set, k)
        accuracies_eul.append(main.evaluate_acc(y, predictions))
        predictions = run_knn(train_set, test_set, k,"manhattan")
        accuracies_man.append(main.evaluate_acc(y, predictions))
        predictions = run_knn(train_set, test_set, k,"minkowski",10)
        accuracies_min.append(main.evaluate_acc(y, predictions))

    plt.plot(k_range, accuracies_eul,color="blue",label = "euclidean")
    plt.plot(k_range, accuracies_man,color="red", label = "manhattan")
    plt.plot(k_range, accuracies_min,color="green",label = "minkowski")
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different K value in KNN with different distance function')
    plt.legend(loc='lower right')
    plt.show()


def plot_p_function(train_set,test_set,y,):
    k_range = np.arange(1, 50)
    accuracies_1 = list()
    accuracies_2 = list()
    accuracies_3 = list()
    for k in k_range:
        predictions = run_knn(train_set, test_set, k, "minkowski",3)
        accuracies_1.append(main.evaluate_acc(y, predictions))
        predictions = run_knn(train_set, test_set, k, "minkowski",5)
        accuracies_2.append(main.evaluate_acc(y, predictions))
        predictions = run_knn(train_set, test_set, k, "minkowski",10)
        accuracies_3.append(main.evaluate_acc(y, predictions))

    plt.plot(k_range, accuracies_1, color="red", label="p=3")
    plt.plot(k_range, accuracies_2, color="green", label="p=5")
    plt.plot(k_range, accuracies_3 ,color="blue",label = "p=10")
    plt.legend(loc='lower right')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different K value in KNN for different p value in minkowski distance function')
    plt.show()


def decision_boundary(x_train,y_train,test,key ="euclidean"):
    x = np.vstack((x_train,test))
    x0v = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 200)
    x1v = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 200)
    x0, x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(), x1.ravel())).T
    train_set = np.hstack((x_train,y_train))
    y_prob_all = run_knn(train_set, x_all, 7,key)
    y_train_prob = np.zeros((y_train.shape[0], 3))
    y_train_index = (y_train/2 -1).astype(int)
    y_train_index = y_train_index.flatten()
    y_train_prob[np.arange(y_train.shape[0]), y_train_index] = 1

    y_prob_all = np.array(y_prob_all)
    y_prob_index =  (y_prob_all/2 -1).astype(int)
    y_prob_index = y_prob_index.flatten()
    y_all = np.zeros((y_prob_all.shape[0], 3))
    y_all[np.arange(y_all.shape[0]), y_prob_index] = 1
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_prob, marker='o', alpha=1)
    plt.scatter(x_all[:, 0], x_all[:, 1], c=y_all, marker='.', alpha=.01)
    plt.xlabel('Clump_Thickness')
    plt.ylabel('Uniformity_of_Cell_Size')
    plt.show()
