import random

import numpy as np

from Assignment2.MultinomialNaiveBayes import MultinomialNaiveBayes
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
from Assignment2.LinearRegression import LinearRegression


import itertools
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


loss = lambda y, yh: np.mean((y - yh) ** 2)


def cross_validation_split(n, n_folds=10):
    # get the number of data samples in each split
    n_val = n // n_folds
    inds = np.random.permutation(n)
    inds = []
    for f in range(n_folds):
        tr_inds = []
        # get the validation indexes
        val_inds = list(range(f * n_val, (f + 1) * n_val))
        # get the train indexes
        if f > 0:
            tr_inds = list(range(f * n_val))
        if f < n_folds - 1:
            tr_inds = tr_inds + list(range((f + 1) * n_val, n))
        # The yield statement suspends function’s execution and sends a value back to the caller
        # but retains enough state information to enable function to resume where it is left off
        yield tr_inds, val_inds


def kFoldCV(model, num_folds, x_train, y_train):
    losses = []
    for f, (tr, val) in enumerate(cross_validation_split(x_train.shape[0], num_folds)):
        model.fit(x_train[tr], y_train[tr])
        predicted = model.predict(x_train[val])
        losses.append(loss(y_train[val], predicted))
    return sum(losses)/num_folds



def evaluate_acc(labels, predictions):
    # labels = labels.flatten()
    # predictions = predictions.flatten()
    accuracy = 0
    for i in range(min(len(labels), len(predictions))):
        if labels[i] == predictions[i]:
            accuracy += 1

    return accuracy / min(len(labels), len(predictions))



def cross_error_plot(alpha_list,err_valid):
    plt.errorbar(alpha_list, err_valid, label='validation')
    plt.legend()
    plt.xlabel('Alpha')
    plt.ylabel('mean squared error')
    plt.show()



def run_MNB(x_train,y_train,X_train_tfidf, x_test):
    num_folds = 5

    alpha_list = np.linspace(0.01, 1.01, 20)
    err_valid = np.zeros((len(alpha_list)))

    for i, alpha in enumerate(alpha_list):
        clf = MultinomialNaiveBayes(alpha)
        err_valid[i] = kFoldCV(clf, num_folds, X_train_tfidf, y_train)

    best = np.argmin(err_valid)
    print(alpha_list[best])
    cross_error_plot(alpha_list, err_valid)
    MNB_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNaiveBayes(alpha_list[best])),
    ])

    MNB_clf.fit(x_train, y_train)
    return MNB_clf.predict( x_test)



def run_LG(x_train,y_train,X_train_tfidf, x_test):
    num_folds = 5

    C_list = np.array( [1.01,1.02,1.03,1.04,1.05,1.06])
    err_valid = np.zeros((len(C_list)))

    for i, c in enumerate(C_list):
        clf = LogisticRegression(C=c,max_iter=500)
        err_valid[i] = kFoldCV(clf, num_folds, X_train_tfidf, y_train)
        print(err_valid)

    best = np.argmin(err_valid)
    LG_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(C=C_list[best],max_iter=500)),
    ])

    LG_clf.fit(x_train, y_train)
    LG_predicted = LG_clf.predict(x_test)
    cross_error_plot(C_list, err_valid)
    return LG_predicted


def run_comparison(x_train,y_train,X_train_tfidf, x_test,y_test):
    MNB_predicted = run_MNB(x_train,y_train,X_train_tfidf, x_test)
    print("test acc MNB: ")
    print(evaluate_acc(MNB_predicted, y_test))
    LG_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(C=1.06)),
    ])

    LG_clf.fit(x_train, y_train)
    LG_predicted = LG_clf.predict(x_test)
    print("test acc LG: ")
    print(evaluate_acc(LG_predicted, y_test))


def load_20_news():
    # reading data
    twenty_train = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))
    twenty_test = fetch_20newsgroups(subset='test', remove=(['headers', 'footers', 'quotes']))
    return twenty_train.data,twenty_train.target,twenty_test.data,twenty_test.target




def load_imdb():
    import os
    files_train_neg = os.listdir('D:/Study/2021 Winter/COMP 551/dataset/train/neg')
    imdb_train = []
    imdb_target = []
    os.chdir('D:/Study/2021 Winter/COMP 551/dataset/train/neg')
    for f in files_train_neg:
        open_file = open(f, "r", encoding="utf8")
        imdb_train.append(open_file.read())
        imdb_target.append(0)

    files_train_pos = os.listdir('D:/Study/2021 Winter/COMP 551/dataset/train/pos')
    os.chdir('D:/Study/2021 Winter/COMP 551/dataset/train/pos')
    for f in files_train_pos:
        open_file = open(f, "r", encoding="utf8")
        imdb_train.append(open_file.read())
        imdb_target.append(1)
    print("fini")


    files_test_neg = os.listdir('D:/Study/2021 Winter/COMP 551/dataset/test/neg')
    imdb_test = []
    imdb_test_target = []
    os.chdir('D:/Study/2021 Winter/COMP 551/dataset/test/neg')
    for f in files_test_neg:
        open_file = open(f,"r",encoding="utf8")
        imdb_test.append(open_file.read())
        imdb_test_target.append(0)


    files_test_pos = os.listdir('D:/Study/2021 Winter/COMP 551/dataset/test/pos')
    os.chdir('D:/Study/2021 Winter/COMP 551/dataset/test/pos')
    for f in files_test_pos:
        open_file = open(f,"r",encoding="utf8")
        imdb_test.append(open_file.read())
        imdb_test_target.append(1)

    print("finished")

    return imdb_train,np.array(imdb_target),imdb_test,imdb_test_target



def run_compare_size(x_train,y_train,x_test,y_test):
    # number of rows of the original matrix
    num_of_xrows = len(x_train)
    num_of_yrows = len(x_train)

    size_list = [0.2, 0.4, 0.6, 0.8]
    acc_list = []
    for size in size_list:
        print(size)
        # 20%(40, 60, 80%) of the num of rows
        size_x = num_of_xrows * size
        size_y = num_of_yrows * size
        # get 20%(40%, 60%, 80%) rows randomly from x_test and y_test
        x_train_sub = random.sample(x_train, int(size_x))
        y_train_sub = y_train[np.random.randint(y_train.shape[0], size=int(size_y))]

        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(x_train_sub)
        tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            #('clf', MultinomialNaiveBayes(0.01)),
            ('clf', LogisticRegression(C=1.03)),
        ])

        clf.fit(x_train_sub, y_train_sub)
        y_test_sub = y_test[np.random.randint(y_test.shape[0], size=int(size_y))]
        #y_test_sub = random.sample(y_test, int(len(y_test)*size))
        acc_list.append(evaluate_acc(clf.predict(random.sample(x_test, int(len(x_test)*size))), y_test_sub))
        #acc_list.append(evaluate_acc(MNB_clf.predict(x_test), y_test))
    plt.errorbar(size_list, acc_list, label='accuracy')
    plt.legend()
    plt.xlabel('Size')
    plt.ylabel('Accuracy')
    plt.show()


def run_linear_regression():
    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearRegression()),
    ])

    clf.fit(x_train, y_train)

    predicted = clf.predict(x_test)
    return predicted



if __name__ == '__main__':
    #x_train,y_train,x_test,y_test = load_imdb()
    x_train,y_train,x_test,y_test = load_20_news()

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(x_train)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    predicted = run_linear_regression()
    #run_compare_size(x_train, y_train, x_test, y_test)
    #predicted = run_MNB(x_train,y_train,X_train_tfidf, x_test)
    print(evaluate_acc(predicted,y_test))
    #run_comparison(x_train, y_train, X_train_tfidf, x_test, y_test)
