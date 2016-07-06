#include "nndemo/NeuralNet.h"

#include <cassert>
#include <cmath>
#include <numeric>

namespace {
float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

unsigned long int max_position(const std::vector<float>& v, int end) {
    unsigned long int result = 0;
    float highest = v[0];
    for (int i = 0; i < end; ++i) {
        if (v[i] > highest) {
            highest = v[i];
            result = i;
        }
    }
    return result;
}
}

nndemo::NeuralNet::~NeuralNet() {}

nndemo::NeuralNet::NeuralNet(NeuralNet&& other)
        : learning_rate(other.learning_rate),
          weights(other.weights),
          nodevalues(other.nodevalues) {}

void nndemo::NeuralNet::train(const std::vector<std::vector<float>>& features,
                              const std::vector<unsigned long int>& labels) {
    // TODO implement
}

std::vector<unsigned long int> nndemo::NeuralNet::predict(const std::vector<std::vector<float>>& features) {
    std::vector<unsigned long int> result;
    for (const std::vector<float>& item : features) {
        forward_propagate(item);
        result.push_back(max_position(nodevalues.back(), nodevalues.back().size()-1));
    }
    return result;
}

void nndemo::NeuralNet::forward_propagate(const std::vector<float>& example) {
    for (int i = 0; i < example.size(); ++i) {
        nodevalues.front()[i] = example[i];
    }
    assert(nodevalues.front().back() == 1.0);
    for (int layer = 1; layer < nodevalues.size(); ++layer) {
        for (int node = 0; node < nodevalues[layer].size()-1; ++node) {
            nodevalues[layer][node] = sigmoid(std::inner_product(std::begin(nodevalues[layer-1]),
                                                                 std::end(nodevalues[layer-1]),
                                                                 std::begin(weights[layer-1][node]),
                                                                 0));
            std::cout << nodevalues[layer][node] << std::endl;
        }
        assert(nodevalues[layer].back() == 1.0);
    }
}
