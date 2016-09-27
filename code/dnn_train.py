#!/usr/bin/env python
# encoding: utf-8

"""
dnn_train.py

Created by Shuailong on 2016-04-16.
Update on 2016-09-27. Switch to Keras.


Trains and Evaluates the NN using keras.

"""

from keras.models import Sequential
from keras.layers import Dense

def run(dataset):

    train_X, train_y = dataset['train']
    dev_X, dev_y = dataset['dev']
    test_X, test_y = dataset['test']

    dimension = len(train_X[0])
    print 'Input dimension: {}'.format(dimension)
    print 'Building model...'
    model = Sequential()
    model.add(Dense(512, input_dim=dimension, activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print 'Training...'
    model.fit(train_X, train_y, nb_epoch=20, batch_size=16)

    print 'Predicting...'
    Acc = model.evaluate(train_X, train_y, batch_size=16)
    print 'Training Data Eval:'
    print 'Acc: {}%'.format(Acc)

    Acc = model.evaluate(dev_X, dev_y, batch_size=16)
    print 'Development Data Eval:'
    print 'Acc: {}%'.format(Acc)

    Acc = model.evaluate(test_X, test_y, batch_size=16)
    print 'Test Data Eval:'
    print 'Acc: {}%'.format(Acc)


def main():
    pass
