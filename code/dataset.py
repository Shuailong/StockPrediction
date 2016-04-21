#!/usr/bin/env python
# encoding: utf-8

"""
dataset.py

Created by Shuailong on 2016-04-06.

Dataset for Machine Learning final project.

"""

import os
import pickle
import csv
import datetime
import re

DATA_PATH = '../data'
CACHE_PATH = '../data/cache'
BLOOMBERG_PATH = '../data/20061020_20131126_bloomberg_news'
REUTERS_PATH = '../data/ReutersNews106521'
SP_PATH = '../data/500'


class Dataset(object):
    '''A class interface to access datasets'''

    def __init__(self, dataset):
        support_dataset = ['Bloomberg', 'Reuters']
        if dataset not in support_dataset:
            raise ValueError('{} is not supported yet. Consider {} instead'.format(
                dataset, support_dataset))
        self.dataset = dataset

        if dataset == support_dataset[0]:
            self.dirname = BLOOMBERG_PATH
        else:
            self.dirname = REUTERS_PATH

        path = os.path.join(DATA_PATH, self.dirname)
        self.dates = os.listdir(path)
        if '.DS_Store' in self.dates:
            self.dates.remove('.DS_Store')

        self.tickers = {}
        with open(os.path.join(DATA_PATH, 'constituents.csv')) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.tickers[row['Symbol']] = {}
                self.tickers[row['Symbol']]['Name'] = row['Name']
                self.tickers[row['Symbol']]['Sector'] = row['Sector']
        self.tickers['GOOG']['Name'] = 'Google Inc.'  # historical reason.

        print '{} dataset initialized. News from {} to {}.'.format(self.dataset, min(self.dates), max(self.dates))

    def get_all_news(self, force=False):
        '''
        Return all news articles
        rtype: {date: List[(title, content)]}
        '''

        CACHE_FILE = os.path.join(CACHE_PATH, self.dataset + '_news.p')
        if not force and os.path.isfile(CACHE_FILE):
            return pickle.load(open(CACHE_FILE, 'rb'))

        print 'No cached found. Building all news from disk...'
        d = {}
        for date in self.dates:
            date_path = os.path.join(DATA_PATH, self.dirname, date)
            articles = []
            for title in os.listdir(date_path):
                if title != '.DS_Store':
                    articles.append(
                        (title, open(os.path.join(date_path, title)).read()))
            key = date.replace('-', '')
            d[key] = articles

        num_news = 0
        for date in d:
            num_news += len(d[date])

        print 'Loaded. {} news articles in {} days.'.format(num_news, len(self.dates))
        pickle.dump(d, open(CACHE_FILE, 'wb'))

        return d

    def get_news_by_company(self, company, force=False):
        '''
        company: ticker name
        Return news articles for a specified company
        rtype: {date: List[(title, content)]}
        '''

        CACHE_FILE = os.path.join(
            CACHE_PATH, self.dataset + '_' + company + '_news.p')
        if not force and os.path.isfile(CACHE_FILE):
            return pickle.load(open(CACHE_FILE, 'rb'))

        print 'No cached found. Building {} news from all news...'.format(company)
        self.all_news = self.get_all_news()

        if company == 'INDEX':
            return self.all_news

        fullname = self.tickers[company]['Name']
        print 'Filtering articles for', company + '/' + fullname, '...'

        company_news = {}
        for date in self.all_news:
            articles = self.all_news[date]
            company_article = []
            for title, content in articles:
                title_l = title.lower()
                content_l = content.lower()
                if fullname.lower() in title_l.lower() or fullname.lower() in content_l.lower():
                    company_article.append((title, content))
            if len(company_article) > 0:
                company_news[date] = company_article

        num_news = 0
        for date in company_news:
            num_news += len(company_news[date])
        print 'Filtering done. {} news articles for {} in {} days.'.format(num_news, company, len(company_news))
        pickle.dump(company_news, open(CACHE_FILE, 'wb'))

        return company_news

    def __repr__(self):
        return self.dataset


class SP500(object):
    '''A class interface to access S&P 500 stock prices'''

    def __init__(self):
        stocks = os.listdir(SP_PATH)
        self.stocks = [stock[:-4]
                       for stock in stocks if stock.endswith('.csv')]

    def get_stock(self, stock):
        '''
        Return stock prices of all dates in database
        rtype: {date: {Open: xx.xx, Close:xx.xx, ...}}
        date format: yyyymmdd
        '''
        if stock not in self.stocks:
            raise ValueError(stock + ' is not in SP500')
        stock_path = os.path.join(SP_PATH, stock + '.csv')

        stock_info = {}
        with open(stock_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                date = row['Date'].replace('-', '')
                stock_info[date] = {}
                stock_info[date]['Open'] = float(row['Open'])
                stock_info[date]['High'] = float(row['High'])
                stock_info[date]['Low'] = float(row['Low'])
                stock_info[date]['Close'] = float(row['Close'])
                stock_info[date]['Volume'] = int(row['Volume'])
                stock_info[date]['Adj Close'] = float(row['Adj Close'])

        return stock_info


class NewsLabels(object):

    ''' A class to construct (news, stock rise/fall) pairs '''

    def __init__(self, company_news, company_stock):

        # train: 2006-10-02 - 2012-06-18
        # dev:   2012-06-19 - 2013-02-21
        # test:  2013-02-22 - 2013-11-12

        self.TRAIN_START = '20061002'
        self.TRAIN_END = '20120619'
        self.DEV_START = '20120619'
        self.DEV_END = '20130222'
        self.TEST_START = '20130222'
        self.TEST_END = '20131113'

        self.company_news = company_news
        self.company_stock = company_stock

    def _date_range(self, start_date, end_date):
        '''
        Return a range of dates from [start_date, end_date)
        rtype: List['yyyymmdd']
        '''
        format = "%Y%m%d"

        start = datetime.datetime.strptime(start_date, format)
        end = datetime.datetime.strptime(end_date, format)

        date_generated = [
            start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
        dates = [date.strftime(format) for date in date_generated]

        return dates

    def _get_data_by_date_range(self, start_date, end_date):
        '''
        Return X, y pair given a specied date range
        rtype: X List[List[String]]
               y List[+1/-1]
        '''

        dates = self._date_range(start_date, end_date)
        X = []
        y = []
        for i in range(len(dates) - 1):
            date = dates[i]
            next_date = dates[i + 1]
            if date in self.company_news:
                titles = [re.sub(r' update\d* ', '', news[0].replace('-', ' '))+'.'
                          for news in self.company_news[date]]
            else:
                continue
            if next_date in self.company_stock:
                diff = self.company_stock[next_date]['Close'] - self.company_stock[next_date]['Open']
                if diff > 0:
                    label = 1
                elif diff < 0:
                    label = -1
                else:
                    # label = 1
                    continue
            else:
                continue

            X.append(titles)
            try:
                y.append(label)
            except:
                print label

        return X, y

    def get_train(self):
        return self._get_data_by_date_range(self.TRAIN_START, self.TRAIN_END)

    def get_dev(self):
        return self._get_data_by_date_range(self.DEV_START, self.DEV_END)

    def get_test(self):
        return self._get_data_by_date_range(self.TEST_START, self.TEST_END)


def main():
    pass


if __name__ == '__main__':
    main()
