#!/usr/bin/env python
# encoding: utf-8

"""
svm_train.py

Created by Shuailong on 2016-04-17.

SVM training.

"""

from math import sqrt
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import PredefinedSplit
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import itertools

def score(true, predict):
    '''
    Return Accuracy and Matthews Correlation Coefficient(MCC).
    '''
    if len(predict) != len(true):
        raise ValueError("Lengths of predicted results != true results: " + str(len(predict)) + ' != ' + str(len(true)))
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(predict)):
        if predict[i] == 1 and true[i] == 1:
            tp += 1
        elif predict[i] == 1 and true[i] == -1:
            fp += 1
        elif predict[i] == -1 and true[i] == -1:
            tn += 1
        else:
            fn += 1
    print 'TP/TN/FP/FN: {} {} {} {}'.format(tp, tn, fp, fn)
    acc = (tp+tn)/float(len(predict))
    try:
        mcc = (tp*tn-fp*fn)/float(sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    except ZeroDivisionError:
        mcc = 0

    return acc, mcc


def run(dataset):

    train_X, train_y = dataset['train']
    dev_X, dev_y = dataset['dev']
    test_X, test_y = dataset['test']

    # param tuning

    param_grid = [
        {
            'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['linear'],
            'gamma': ['auto']
            },
        {
            'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['rbf'],
            'gamma': ['auto', 0.001, 0.01, 0.1, 1]
            }
        ]

    best_params = {}
    best_accuracy = 0

    clf = SVC(verbose=False)
    for d in param_grid:
        keys = d.keys()
        for v1 in d[keys[0]]:
            for v2 in d[keys[1]]:
                for v3 in d[keys[2]]:
                    params = {keys[0]: v1, keys[1]: v2, keys[2]: v3}
                    print 'Params:', params
                    clf.set_params(**params)
                    clf.fit(train_X, train_y)
                    acc = clf.score(dev_X, dev_y)
                    print 'Acc:', acc
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_params = params
    clf.set_params(**best_params)
    clf.fit(train_X, train_y)
    print best_params
    print 'Predicting...'
    predict_y = clf.predict(train_X)
    Acc, MCC = score(train_y, predict_y)
    print 'Training Data Eval:'
    print 'Acc: {}%\tMCC: {}%'.format(round(Acc*100, 2), round(MCC*100, 2))

    predict_y = clf.predict(dev_X)
    Acc, MCC = score(dev_y, predict_y)
    print 'Development Data Eval:'
    print 'Acc: {}%\tMCC: {}%'.format(round(Acc*100, 2), round(MCC*100, 2))

    predict_y = clf.predict(test_X)
    Acc, MCC = score(test_y, predict_y)
    print 'Test Data Eval:'
    print 'Acc: {}%\tMCC: {}%'.format(round(Acc*100, 2), round(MCC*100, 2))
