#!/usr/bin/env python
# encoding: utf-8

"""
featureextractor.py

Created by Shuailong on 2016-04-06.

Feature extractor for final project.

"""

from openIENER.API import mineOneSentence
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import os

from dataset import CACHE_PATH


class BoW(object):
    '''
    Bag of Words features extractor.

    '''
    def __init__(self, X):
        '''
        X: training data sentences.
        type X: List[List[String]]
        '''
        X = [''.join(x) for x in X]
        self.vectorizer = TfidfVectorizer(decode_error='ignores', stop_words='english',)
        self.vectorizer.fit(X)

    def get_feature(self, X):
        '''
        X: a sentence to be extracted.
        type X: List[String]
        '''
        X = [''.join(x) for x in X]
        return self.vectorizer.transform(X).toarray()


class Event(object):
    def __init__(self, X, force=False):
        '''
        Build a total events list from dataset.
        X: List[List[String]]
        '''

        CACHE_FILE = os.path.join(CACHE_PATH, 'all_events.p')
        if not force and os.path.isfile(CACHE_FILE):
            self.events_lists = pickle.load(open(CACHE_FILE, 'rb'))
        else:
            print 'No cached found. Building all events from data...'
            events_lists = set()
            for sents in X:
                for sent in sents:
                    events = mineOneSentence(sent.decode('utf-8'))
                    for event in events:
                        sub, action, obj, _ = event
                        events_lists.add(sub)
                        events_lists.add(action)
                        events_lists.add(obj)
                        events_lists.add(sub+action)
                        events_lists.add(action+obj)
                        events_lists.add(sub+action+obj)
            self.events_lists = list(events_lists)
            pickle.dump(self.events_lists, open(CACHE_FILE, 'wb'))

        print 'Events list loaded.'
        self.num_events = len(self.events_lists)
        print 'Number of total events: {}'.format(self.num_events)

    def get_feature(self, X):
        '''
        Event feature from sentences.
        X: List[List[String]]
        rtype: List[[0|1]]
        '''
        features = []
        for sents in X:
            x_events = set()
            for sent in sents:
                events = mineOneSentence(sent.decode('utf-8'))
                for event in events:
                    sub, action, obj, _ = event
                    x_events.add(sub)
                    x_events.add(action)
                    x_events.add(obj)
                    x_events.add(sub+action)
                    x_events.add(action+obj)
                    x_events.add(sub+action+obj)

            feature = np.zeros(self.num_events, dtype=np.int8)
            for i, event in enumerate(self.events_lists):
                feature[i] = 1 if event in x_events else 0
            features.append(feature)

        return features


def main():
    pass

if __name__ == '__main__':
    main()
