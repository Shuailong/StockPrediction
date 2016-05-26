#!/usr/bin/env python
# encoding: utf-8

"""
main.py

Created by Shuailong on 2016-04-07.

Main entry for Machine Learning final project.

"""

from predictor import Predictor

from time import time


def main():
    start_time = time()

    predictor = Predictor(company='INDEX', dataset='Bloomberg', classifier='NN', feature_type='Event')
    predictor.run()

    # companies = ['GOOG', 'BA', 'WMT', 'INDEX']
    # # datasets = ['Bloomberg', 'Reuters']
    # classifiers = ['SVM', 'NN']
    # feature_types = ['BoW', 'Event']

    # for company in companies:
    #     for classifier in classifiers:
    #         for feature_type in feature_types:
    #             predictor = Predictor(company=company, dataset='Bloomberg', classifier=classifier, feature_type=feature_type)
    #             print '##############################'
    #             print 'Company:', company
    #             print 'Classifier:', classifier
    #             print 'Feature type:', feature_type

    #             predictor.run()

    print '----------' + str(round(time() - start_time, 2)) + ' seconds.---------------'

if __name__ == '__main__':
    main()
