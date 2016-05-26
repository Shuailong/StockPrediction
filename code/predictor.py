#!/usr/bin/env python
# encoding: utf-8

"""
predictor.py
Created by Shuailong on 2016-04-12.

Predictor class for Machine Learning final project.

"""

from dataset import *
from feature_extractor import BoW, Event
import svm_train
import dnn_train

import pickle
import os
import sys


class Predictor(object):
    def __init__(self, company='GOOG', dataset='Bloomberg', classifier='SVM', feature_type='BoW'):

        self.company = company
        self.dataset = dataset
        if classifier not in ['SVM', 'NN']:
            raise ValueError('Classifier ' + classifier + ' is not supported!')
        self.classifier = classifier
        if feature_type not in ['BoW', 'Event']:
            raise ValueError('Feature type ' + feature_type + ' is not supported!')
        self.feature_type = feature_type

    def run(self):
        # Get news for the company
        print
        news = Dataset(self.dataset)
        print 'Getting news for {} in {} dataset...'.format(self.company, news)
        company_news = news.get_news_by_company(self.company, force=False)  # {date: List[(title, content)]}

        # Get stock prices for the company
        sp = SP500()
        company_stock = sp.get_stock(self.company)  # {date: {Open: xx.xx, Close:xx.xx, ...}}

        # Construct X(string list) and y(label) for train, dev and test
        news_labels = NewsLabels(company_news, company_stock)
        train_X, train_y = news_labels.get_train()
        dev_X, dev_y = news_labels.get_dev()
        test_X, test_y = news_labels.get_test()

        train_pos, train_neg = train_y.count(1), train_y.count(-1)
        dev_pos, dev_neg = dev_y.count(1), dev_y.count(-1)
        test_pos, test_neg = test_y.count(1), test_y.count(-1)

        # DEBUG
        CACHE_FILE = os.path.join(CACHE_PATH, self.dataset + '_' + self.company + '_' + 'feature.p')
        dataset = pickle.load(open(CACHE_FILE, 'rb'))
        print 'training...'
        dnn_train.run(dataset)
        sys.exit(0)

        # Extract features
        if self.feature_type == 'BoW':
            feature_extractor = BoW(train_X)
        elif self.feature_type == 'Event':
            feature_extractor = Event(train_X)
        else:
            pass

        train_X = feature_extractor.get_feature(train_X)
        dev_X = feature_extractor.get_feature(dev_X)
        test_X = feature_extractor.get_feature(test_X)

        print
        print 'Number of train examples :\t{}, POS: \t{}, NEG: \t{}'.format(len(train_X), train_pos, train_neg)
        print 'Number of dev examples :\t{}, POS: \t{}, NEG: \t{}'.format(len(dev_X), dev_pos, dev_neg)
        print 'Number of test examples :\t{}, POS: \t{}, NEG: \t{}'.format(len(test_X), test_pos, test_neg)
        print
        print 'Baseline(majority vote) accuracy: '
        print 'Train: {}%'.format(round(max(train_pos, train_neg)/float(train_pos+train_neg)*100, 2))
        print 'Dev: {}%'.format(round(max(dev_pos, dev_neg)/float(dev_pos+dev_neg)*100, 2))
        print 'Test: {}%'.format(round(max(test_pos, test_neg)/float(test_pos+test_neg)*100, 2))
        print

        if self.classifier == 'SVM':
            dataset = {'train': (train_X, train_y), 'dev': (dev_X, dev_y), 'test': (test_X, test_y)}
            print 'Training...'

            svm_train.run(dataset)
        elif self.classifier == 'NN':
            train_y = [1 if y == 1 else 0 for y in train_y]
            dev_y = [1 if y == 1 else 0 for y in dev_y]
            test_y = [1 if y == 1 else 0 for y in test_y]
            dataset = {'train': (train_X, train_y), 'dev': (dev_X, dev_y), 'test': (test_X, test_y)}

            # Save training data
            CACHE_FILE = os.path.join(CACHE_PATH, self.dataset + '_' + self.company + '_' + 'feature.p')
            pickle.dump(dataset, open(CACHE_FILE, 'wb'))
            print 'Training...'
            dnn_train.run(dataset)
        else:
            pass


def main():
    pass

if __name__ == '__main__':
    main()
