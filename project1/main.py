import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import knn
import decision_tree as dt

def evaluate_acc(labels, predictions):
    #labels = labels.flatten()
    #predictions = predictions.flatten()
    accuracy = 0
    for i in range(min(len(labels), len(predictions))):
        if labels[i] == predictions[i]:
            accuracy += 1

    return accuracy / min(len(labels), len(predictions))


if __name__ == '__main__':
    # read and lean the data
    df = pd.read_csv('breast_cancer_wisconsin.csv')
    #df = pd.read_csv('hepatitis.csv')
    df = df[~df.eq('?').any(1)]

    # slipt the data set into train and test set
    train_set = df.sample(frac=0.8, random_state=100)  # random state is a seed
    test_set = df.drop(train_set.index)

    # if the csv file is about the breast cancer
    if (df.iloc[0].index[0] == "id"):

        # Below is the last column of the test_set, which is the real value of class.(y)
        y = test_set.iloc[:, -1:]
        # Below is the test data set which is obtained by deleting the last column of the test_set
        test_set = test_set.iloc[:, :-1]

        # Another thing we need to consider is to drop the first row of our dataset
        # since we cannot do the numerical operations on strings
        y = y.iloc[1:]
        test_set = test_set.iloc[1:]
        train_set = train_set.iloc[1:]

        # when it comes to the breast_cancer data, it makes no sense to take the id into consideration we do machine learning
        # So we delete the first column of test_set and train_set
        test_set = test_set.loc[:, test_set.columns != 'id']
        train_set = train_set.loc[:, train_set.columns != 'id']

    else:

        # Below is the first column of the test_set, which is the real value of class.(y)
        y = test_set[test_set.columns[0]]
        # Below is the test data set which is obtained by deleting the last column of the test_set
        test_set = test_set.loc[:, test_set.columns != 'Class']

        # Another thing we need to consider is to drop the first row of our dataset
        # since we cannot do the numerical operations on strings
        y = y.iloc[1:]
        test_set = test_set.iloc[1:]
        cols = train_set.columns.tolist()
        cols = cols[1:] + cols[:1]
        train_set = train_set[cols]
        # train_set = train_set.iloc[1:]

        # note that our methods are using the numpy array,
        # so we need to change  our dataframe into the numpy array.
    train_set = train_set.to_numpy()
    test_set = test_set.to_numpy()
    y = y.to_numpy()

    #dt.plot_cost_function(train_set, test_set, y)
    #dt.plot_cost_function(train_set, test_set, y, )
    #knn.plot_p_function(train_set, test_set, y)
    #knn.plot_distance_function(train_set, test_set, y)

    #decision boundary code
    test_x = test_set[:,0:2]
    train_x = train_set[:, 0:2]
    train_y = train_set[:, -1:]
    #dt.decision_boundary(train_x,train_y,test_x,"miss_rate")
    knn.decision_boundary(train_x, train_y, test_x,key = "minkowski")

