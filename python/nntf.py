"""Standard Multilayer Perceptron using TensorFlow

After completing the numpy version, I got interested in using TensorFlow.  This
is the result.
"""
import argparse

import numpy as np
import tensorflow as tf


TRAIN_PERC = 0.6


#pylint:disable-msg=too-many-instance-attributes
class NeuralNet():
    """Implementation of multilayer perceptron"""

    def __init__(self, arch):
        """Build network according to arch"""
        self.learning_rate = float(arch[0])
        self.values = [tf.placeholder(tf.float32, shape=[None, int(arch[1])])]
        self.weights = []
        self.biases = []
        for pos in range(1, len(arch)-1):
            num = int(arch[pos])
            nextnum = int(arch[pos+1])
            self.weights.append(
                tf.Variable(tf.random_uniform([num, nextnum],
                                              -0.05,
                                              0.05)))
            # output layer doesn't have bias node
            self.biases.append(tf.Variable(tf.random_uniform([nextnum],
                                                             -0.05,
                                                             0.05)))
            self.values.append(
                tf.sigmoid(
                    tf.matmul(self.values[-1], self.weights[-1]) +\
                    self.biases[-1]))
        self.targets_count = int(arch[-1])
        self.targets = tf.placeholder(tf.float32, shape=[None, int(arch[-1])])
        self.loss = tf.reduce_mean(tf.square(self.values[-1] - self.targets))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=self.global_step)
        self.prediction = tf.argmax(self.values[-1], 1)

    def train(self, sess, featureses, labels):
        """Train self on featureses according to labels"""
        truths = []
        for cur in range(self.targets_count):
            tmp = np.zeros(self.targets_count)
            tmp[cur] = 1
            truths.append(tmp)
        for _ in range(100):
            #pylint:disable-msg=consider-using-enumerate
            for pos in range(len(labels)):
                feed_dict = {
                    self.values[0]: featureses[pos].reshape(
                        1, len(featureses[pos])),
                    self.targets: truths[labels[pos]].reshape(
                        1, self.targets_count)}
                _, error = sess.run(
                    [self.train_op, self.loss], feed_dict=feed_dict)
                print(error)

    def predict(self, sess, featureses):
        """Predict classes based on features"""
        result = []
        #pylint:disable-msg=consider-using-enumerate
        for pos in range(len(featureses)):
            feed_dict = {self.values[0]: featureses[pos].reshape(
                1, len(featureses[pos]))}
            result.append(sess.run(self.prediction, feed_dict=feed_dict))
        return result


def _parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(
        description='Run a multilayer perceptron via vector operations')
    parser.add_argument('data', help='file containing data')
    parser.add_argument('arch', help='file containing architecture information')
    return parser.parse_args()


def _get_data(filename):
    """Extract data from file"""
    features = []
    labels = []
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                items = line.split(',')
                features.append([float(num) for num in items[:-1]])
                labels.append(int(items[-1]))
    return np.array(features), np.array(labels)


def _parse_arch(filename):
    """Extract architecture information from file"""
    arch = []
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                arch.append(line)
    return arch


def _split_indices(cutoff, count):
    """Split indices"""
    result = list(range(count))
    np.random.shuffle(result)
    threshold = int(cutoff * count)
    return np.array(result[:threshold]), np.array(result[threshold:])


def _run():
    """Run the code"""
    args = _parse_args()
    features, labels = _get_data(args.data)
    train_indices, test_indices = _split_indices(TRAIN_PERC, len(labels))
    with tf.Graph().as_default(), tf.Session() as sess:
        model = NeuralNet(_parse_arch(args.arch))
        init = tf.initialize_all_variables()
        sess.run(init)
        model.train(sess, features[train_indices], labels[train_indices])
        predictions = model.predict(sess, features[test_indices])
        #pylint:disable-msg=consider-using-enumerate
    for pos in range(len(test_indices)):
        print(labels[test_indices[pos]], predictions[pos])


if __name__ == '__main__':
    _run()
