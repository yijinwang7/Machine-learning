#!/usr/bin/python

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot  as plt
import main

def miss_rate(data):
    if len(data) == 0:
        return 0
    values = [row[-1] for row in data]
    prediction = find_most_frequent(data)
    return (len(values) - values.count(prediction)) / len(values)

def entropy(data):
    if len(data) == 0:
        return 0
    values = [row[-1] for row in data]

    elements = set(values)
    entropy = 0.
    for v in elements:
        p_v = (values.count(v)) / len(values)
        entropy -= p_v *math.log(p_v,math.e)
    return entropy

def gini(data):
    if len(data) == 0:
        return 0
    values = [row[-1] for row in data]
    elements = set(values)
    gini = 0.
    for v in elements:
        p_v = (values.count(v)) / len(values)
        gini += p_v *(1-p_v)
    return gini




def greedy_test(data,key = "miss_rate"):
    best_cost = float('inf')
    best_split = 0
    best_value = 0
    best_left = []
    best_right = []
    for i in range(data.shape[1] - 1):
        for value in data:
            # split in to left and right node
            split_test = value[i]
            data_left = []
            data_right = []
            for instances in data:
                if instances[i] < split_test:
                    data_left.append(instances)
                else:
                    data_right.append(instances)
                # check for optimal split cost
            c_left = miss_rate(data_left)
            c_right = miss_rate(data_right)
            if key == "entropy":
                c_left = entropy(data_left)
                c_right = entropy(data_right)
            elif key == "gini":
                c_left = gini(data_left)
                c_right = gini(data_right)
            split_cost = len(data_left) / len(data) * c_left + len(data_right) / len(data) * c_right
            if split_cost < best_cost:
                best_split = i
                best_value = split_test
                best_cost = split_cost
                best_left = data_left
                best_right = data_right
    return best_left, best_right, best_cost, best_split,best_value


def find_most_frequent(data):
    values = [row[-1] for row in data]
    prediction = max(set(values), key=values.count)
    return prediction




class Node:

    def __init__(self, depth):
        self.left = None
        self.right = None
        self.depth = depth
        self.splitting_dimension = 0
        self.splitting_value = 0
        self.is_leaf = False
        self.prediction = None

    def fit_recursive(self, data, max_depth,key = "miss_rate"):
        left, right, best_cost,best_split,best_value = greedy_test(data,key)
        left = np.asarray(left)
        right = np.asarray(right)
        cost = miss_rate(data)
        if key == "entropy":
            cost = entropy(data)
        elif key == "gini":
            cost = gini(data)
        if self.depth + 1 > max_depth or len(right) == 1 or len(left) == 1 or best_cost < 0.0001 or (
                cost - best_cost) < 0.0001:
            self.is_leaf = True
            self.prediction = find_most_frequent(data)
            return self
        else:
            node_left = Node(self.depth + 1)
            node_right = Node(self.depth + 1)
            self.splitting_dimension = best_split
            self.splitting_value = best_value
            self.left = node_left.fit_recursive(left, max_depth,key)
            self.right = node_right.fit_recursive(right, max_depth,key)
            return self

    def predict(self, target):
        if self.is_leaf:
            return self.prediction
        else:
            if target[self.splitting_dimension] < self.splitting_value:
                return self.left.predict(target)
            else:
                return self.right.predict(target)



class DecisionTree:

    def __init__(self):
        self.head = Node(0)
        self.max_depth = 0

    def fit(self, training_data, max_depth,key):
        self.max_depth = max_depth
        self.head.fit_recursive(training_data, max_depth,key)

    def predict(self, targets, key = "missrate"):
        predictions = list()
        for target in targets:
            target = target.flatten()
            target_prediction = self.head.predict(target)
            predictions.append(target_prediction)
        return predictions

def run_decision_tree(train_set,test_set,max_depth,key ="miss_rate"):
    ### run Decision_tree:
    dt = DecisionTree()
    dt.fit(train_set, max_depth,key)  # train the data
    y_head = dt.predict(test_set)  # use the test data set to get the prediction y^
    return y_head


def plot_depth(train_set, test_set,y):
    d_range = np.arange(1,10)
    accuracies = list()
    for k in d_range:
        predictions = run_decision_tree(train_set,test_set,k)
        accuracies.append(main.evaluate_acc(y,predictions))
    plt.plot(d_range, accuracies)
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different max_depth value in decision tree ')
    plt.show()


def plot_cost_function(train_set,test_set,y,):
    d_range = np.arange(1,15)
    accuracies_miss = list()
    accuracies_en= list()
    accuracies_gini = list()
    for k in d_range:
        predictions = run_decision_tree(train_set,test_set,k)
        accuracies_miss.append(main.evaluate_acc(y,predictions))
        predictions = run_decision_tree(train_set,test_set,k,"entropy")
        accuracies_en.append(main.evaluate_acc(y,predictions))
        predictions = run_decision_tree(train_set,test_set,k,"gini")
        accuracies_gini.append(main.evaluate_acc(y,predictions))

    plt.plot(d_range, accuracies_miss,color="blue",label = "miss rate")
    plt.plot(d_range, accuracies_en,color="red",label = "entropy")
    plt.plot(d_range, accuracies_gini,color="green",label = "gini")
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different max_depth value in decision tree with different cost function')
    plt.legend(loc='lower right')

    plt.show()


def decision_boundary(x_train,y_train,test,key="entropy"):
    x = np.vstack((x_train,test))
    x0v = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 200)
    x1v = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 200)
    x0, x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(), x1.ravel())).T
    train_set = np.hstack((x_train,y_train))
    y_prob_all = run_decision_tree(train_set, x_all, 200,key)
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
