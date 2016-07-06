#include "nndemo/NeuralNet.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

namespace {
float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

unsigned long int max_position(const std::vector<float>& v) {
    unsigned long int result = 0;
    float highest = v[0];
    for (int i = 0; i < v.size(); ++i) {
        // std::cout << v[i] << " ";
        if (v[i] > highest) {
            highest = v[i];
            result = i;
        }
    }
    // std::cout << std::endl;
    return result;
}

std::vector<float> build_truth_vector(unsigned long int size, unsigned long int pos) {
    std::vector<float> result(size, 0.0);
    result[pos] = 1.0;
    return result;
}

std::vector<float> elementwise_subtract(const std::vector<float>& v,
                                        const std::vector<float>& u) {
    std::vector<float> result;
    for (int i = 0; i < v.size(); ++i) {
        result.push_back(v[i] - u[i]);
    }
    return result;
}

std::vector<float> elementwise_multiply(const std::vector<float>& v,
                                        const std::vector<float>& u) {
    std::vector<float> result;
    for (int i = 0; i < v.size(); ++i) {
        result.push_back(v[i] * u[i]);
    }
    return result;
}

float error_function(const std::vector<float>& v) {
    float result = 0.0;
    for (auto diff : v) {
        result += diff * diff;
    }
    return result;
}

float calculate_error(const std::vector<float>& outputs,
                      unsigned long int pos) {
    return error_function(elementwise_subtract(build_truth_vector(outputs.size(),
                                                                  pos),
                                               outputs));
}
}

nndemo::NeuralNet::~NeuralNet() {}

nndemo::NeuralNet::NeuralNet(NeuralNet&& other)
        : learning_rate(other.learning_rate),
          weights(other.weights),
          nodevalues(other.nodevalues),
          deltas(other.deltas) {}

void nndemo::NeuralNet::train(const std::vector<std::vector<float>>& features,
                              const std::vector<unsigned long int>& labels) {
    const float EPSILON = 0.001;
    std::vector<std::vector<float>> oneses;
    for (auto& layer : nodevalues) {
        oneses.emplace_back(std::vector<float>(layer.size(), 1.0));
    }
    std::vector<std::vector<float>> truths;
    for (int i = 0; i < nodevalues.back().size(); ++i) {
        truths.emplace_back(build_truth_vector(nodevalues.back().size(), i));
    }
    //float prev_error = std::numeric_limits<float>::infinity();
    float cur_error = std::numeric_limits<float>::max();
    //while (abs(cur_error - prev_error) > EPSILON) {
        //prev_error = cur_error;
    for (int iter = 0; iter < 100; ++iter) {
        cur_error = 0.0;
        for (int i = 0; i < features.size(); ++i) {
            forward_propagate(features[i]);
            cur_error += calculate_error(nodevalues.back(), labels[i]);
            // backprop
            deltas.back() = elementwise_multiply(elementwise_subtract(nodevalues.back(),
                                                                      truths[labels[i]]),
                                                 elementwise_multiply(nodevalues.back(),
                                                                      elementwise_subtract(oneses.back(),
                                                                                           nodevalues.back())));
            for (int j = deltas.size()-2; j >= 1; --j) {
                // input layer nodes don't need to have a delta
                std::vector<float> weightederror;
                for (int k = 0; k < weights[j].size(); ++k) {
                    weightederror.push_back(std::inner_product(std::begin(deltas[j+1]),
                                                               std::end(deltas[j+1]),
                                                               std::begin(weights[j][k]),
                                                               0.0f));
                }
                // bias weights have a delta of 0, since bias nodes always output 1
                deltas[j] = elementwise_multiply(weightederror,
                                                 elementwise_multiply(nodevalues[j],
                                                                      elementwise_subtract(oneses[j],
                                                                                           nodevalues[j])));
                /*
                std::cout << j << "\t" << deltas[j].size() << "\t";
                for (auto d : deltas[j]) {
                    std::cout << d << " ";
                }
                std::cout << std::endl;
                */
            }
            for (int layer = 0; layer < weights.size(); ++layer) {
                for (int fromnode = 0; fromnode < weights[layer].size(); ++fromnode) {
                    for (int tonode = 0; tonode < weights[layer][fromnode].size(); ++tonode) {
                        float weight_change = learning_rate * deltas[layer+1][tonode] * nodevalues[layer][fromnode];
                        // std::cout << weight_change << " ";
                        weights[layer][fromnode][tonode] -= weight_change;
                    }
                }
            }
            // std::cout << std::endl;
        }
        std::cout << cur_error << std::endl;
    }
}

std::vector<unsigned long int> nndemo::NeuralNet::predict(const std::vector<std::vector<float>>& features) {
    std::vector<unsigned long int> result;
    for (const std::vector<float>& item : features) {
        forward_propagate(item);
        result.push_back(max_position(nodevalues.back()));
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
            std::vector<float> relevant_weights;
            for (int i = 0; i < nodevalues[layer-1].size()-1; ++i) {
                relevant_weights.push_back(weights[layer-1][i][node]);
            }
            nodevalues[layer][node] = sigmoid(std::inner_product(std::begin(nodevalues[layer-1]),
                                                                 std::end(nodevalues[layer-1]),
                                                                 std::begin(relevant_weights),
                                                                 0.0f));
            // std::cout << nodevalues[layer][node] << std::endl;
        }
        // output layer doesn't have bias term
        if (layer < nodevalues.size()-1) assert(nodevalues[layer].back() == 1.0);
    }
}
