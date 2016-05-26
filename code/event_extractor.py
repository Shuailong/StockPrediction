#!/usr/bin/env python
# encoding: utf-8

"""
featureextractor.py

Created by Shuailong on 2016-05-18.

Extract events tuple from news, given stock ticker.

"""


from dataset import *
from feature_extractor import Event


class EventExtractor(object):
    def __init__(self, company='INDEX', dataset='Bloomberg'):
        self.company = company
        self.dataset = dataset

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

        # Extract features
        feature_extractor = Event(train_X)

        train_X = feature_extractor.get_feature(train_X)
        dev_X = feature_extractor.get_feature(dev_X)
        test_X = feature_extractor.get_feature(test_X)

        print train_X


def main():
    ee = EventExtractor()
    ee.run()


if __name__ == '__main__':
    main()
