#!/usr/bin/env python
# encoding: utf-8

"""
dnn_train.py

Created by Shuailong on 2016-04-16.

Trains and Evaluates the NN using a feed dictionary.

Modified from tensorflow minist example.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dnn_model

import time
from six.moves import xrange
import tensorflow as tf


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 1024, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('summaries_dir', '/tmp/dnn_logs', 'Summaries directory')
flags.DEFINE_integer('max_iters', 3, 'Number of iterations of the train dataset')


def placeholder_inputs(batch_size, n_features):
    """
    Generate placeholder variables to represent the input tensors.
    """

    features_placeholder = tf.placeholder(
        tf.float32, shape=(batch_size, n_features))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return features_placeholder, labels_placeholder


def fill_feed_dict(dataset, features_pl, labels_pl):
    """
    Fills the feed_dict for training the given step.
    """

    features_feed, labels_feed = dataset

    feed_dict = {
        features_pl: features_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            features_placeholder,
            labels_placeholder,
            dataset):
    """Runs one evaluation against the full epoch of data.

    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      featuress_placeholder: The features placeholder.
      labels_placeholder: The labels placeholder.
      dataset: The set of features and labels to evaluate.
    """

    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = len(dataset) // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(dataset[step],
                                   features_placeholder,
                                   labels_placeholder)

        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    Acc = true_count / num_examples
    MCC = 0

    print('Acc: {}%\tMCC: {}%'.format(round(Acc*100, 2), round(MCC*100, 2)))


def run(dataset):
    """Train model for a number of steps."""

    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        tf.gfile.MakeDirs(FLAGS.summaries_dir)

    train = dataset['train']
    dev = dataset['dev']
    test = dataset['test']

    n_features = len(train[0][0])
    n_data = len(train[0])

    train_pairs = []
    for X, y in zip(train[0], train[1]):
        train_pairs.append(([X], [y]))

    dev_pairs = []
    for X, y in zip(dev[0], dev[1]):
        dev_pairs.append(([X], [y]))

    test_pairs = []
    for X, y in zip(test[0], test[1]):
        test_pairs.append(([X], [y]))

    with tf.Graph().as_default():
        features_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size, n_features)

        logits = dnn_model.inference(features_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        loss = dnn_model.loss(logits, labels_placeholder)
        train_op = dnn_model.training(loss, FLAGS.learning_rate)
        eval_correct = dnn_model.evaluation(logits, labels_placeholder)

        summary_op = tf.merge_all_summaries()

        sess = tf.Session()
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        init = tf.initialize_all_variables()
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir, sess.graph)

        step = 0
        for iter_ in xrange(FLAGS.max_iters):
            print('{}th iteration'.format(iter_))
            for step_ in xrange(n_data):
                print('{}th data'.format(step_))
                feed_dict = fill_feed_dict(train_pairs[step_],
                                           features_placeholder,
                                           labels_placeholder)

                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                step += 1
                if step % 10 == 0:
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                features_placeholder,
                labels_placeholder,
                train_pairs)
        print('Development Data Eval:')
        do_eval(sess,
                eval_correct,
                features_placeholder,
                labels_placeholder,
                dev_pairs)
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                features_placeholder,
                labels_placeholder,
                test_pairs)


def main():
    pass
