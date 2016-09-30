#!/usr/bin/env python
# encoding: utf-8

"""
dnn_train.py

Created by Shuailong on 2016-04-16.
Update on 2016-09-27. Switch to Keras.


Trains and Evaluates the NN using keras.

"""

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout

def run(dataset):

    batch_size = 16
    nb_epoch = 20

    train_X, train_y = dataset['train']
    dev_X, dev_y = dataset['dev']
    test_X, test_y = dataset['test']

    
    print('train_X shape:', train_X.shape)

    print('Building model...')
    
    model = Sequential()
    model.add(Dense(1024, input_dim=train_X.shape[1], activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    model.fit(train_X, train_y, 
            batch_size=batch_size, nb_epoch=nb_epoch,
            verbose=1, validation_data=(dev_X, dev_y))
    
    score = model.evaluate(test_X, test_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def main():
    pass
