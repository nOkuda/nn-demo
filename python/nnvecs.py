"""Standard Multilayer Perceptron using vector operations

After coding up the multilayer perceptron in C++, I began to wonder how to
formulate the algorithm using vector operations.  This is the result of my
curiosity.
"""
import argparse

import numpy as np
from scipy.special import expit


TRAIN_PERC = 0.6


class NeuralNet():
    """Implementation of multilayer perceptron"""

    def __init__(self, arch):
        """Build network according to arch"""
        self.learning_rate = float(arch[0])
        self.values = []
        self.deltas = []
        self.weights = []
        for pos in range(1, len(arch)-1):
            num = int(arch[pos])
            # account for bias node
            self.values.append(np.ones(num+1))
            self.deltas.append(np.ones(num+1))
            self.weights.append((np.random.rand(num+1, int(arch[pos+1]))-0.5)*0.1)
        # output layer doesn't have bias node
        self.values.append(np.ones(int(arch[-1])))
        self.deltas.append(np.ones(int(arch[-1])))

    def train(self, featureses, labels):
        """Train self on featureses according to labels"""
        truths = []
        for cur in range(len(self.values[-1])):
            tmp = np.zeros(len(self.values[-1]))
            tmp[cur] = 1
            truths.append(tmp)
        oneses = []
        for layer in self.values:
            oneses.append(np.ones(len(layer)))
        for _ in range(100):
            #pylint:disable-msg=consider-using-enumerate
            for pos in range(len(labels)):
                self._forward_propagate(featureses[pos])
                error = 0.5 * np.square(truths[labels[pos]] - self.values[-1]).sum()
                print(error)
                # backprop
                self.deltas[-1] = (self.values[-1] - truths[labels[pos]]) \
                    * self.values[-1] * (oneses[-1] - self.values[-1])
                for layer in range(len(self.deltas)-2, 0, -1):
                    tmp = self.values[layer] * (oneses[layer] - self.values[layer])
                    tmp[-1] = 1.0
                    self.deltas[layer] = np.dot(
                        self.weights[layer], self.deltas[layer+1]) * tmp
                for layer in range(len(self.weights)-1):
                    self.weights[layer] -= self.learning_rate \
                            * np.outer(self.values[layer],
                                       self.deltas[layer+1][:-1])
                self.weights[-1] -= self.learning_rate \
                        * np.outer(self.values[-2],
                                   self.deltas[-1])

    def predict(self, featureses):
        """Predict classes based on features"""
        result = []
        #pylint:disable-msg=consider-using-enumerate
        for pos in range(len(featureses)):
            self._forward_propagate(featureses[pos])
            result.append(np.argmax(self.values[-1]))
        return result

    def _forward_propagate(self, features):
        """Propagate forward"""
        self.values[0][:-1] = features
        for layer in range(len(self.weights)):
            if layer < len(self.weights)-1:
                self.values[layer+1][:-1] = expit(
                    np.dot(self.values[layer], self.weights[layer]))
            else:
                self.values[layer+1] = expit(
                    np.dot(self.values[layer], self.weights[layer]))


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
    model = NeuralNet(_parse_arch(args.arch))
    model.train(features[train_indices], labels[train_indices])
    predictions = model.predict(features[test_indices])
    #pylint:disable-msg=consider-using-enumerate
    for pos in range(len(test_indices)):
        print(labels[test_indices[pos]], predictions[pos])


if __name__ == '__main__':
    _run()
