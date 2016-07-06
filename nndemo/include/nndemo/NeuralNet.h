#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <string>

#include "nndemo/FileReader.h"

namespace nndemo {
class NeuralNet {
public:
    template<class URNG> NeuralNet(const std::vector<std::string>& parsed, URNG& rng) {
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        learning_rate = stof(parsed.front());
        for (int i = 1; i < parsed.size()-1; ++i) {
            weights.emplace_back(std::vector<std::vector<float>>());
            // account for bias
            int curcount = stoi(parsed[i]) + 1;
            nodevalues.emplace_back(std::vector<float>(curcount));
            deltas.emplace_back(std::vector<float>(curcount));
            *(nodevalues.back().rbegin()) = 1.0;
            int nextcount = stoi(parsed[i+1]);
            for (int j = 0; j < curcount; ++j) {
                weights.back().push_back(std::vector<float>());
                for (int k = 0; k < nextcount; ++k) {
                    // initialize weights randomly
                    weights.back().back().push_back((distribution(rng) - 0.5) * 0.1);
                }
            }
        }
        // output layer doesn't have weights out from it
        int outputscount = stoi(parsed.back());
        nodevalues.emplace_back(std::vector<float>(outputscount));
        deltas.emplace_back(std::vector<float>(outputscount));
        std::cout << "weights" << std::endl;
        for (auto& layer : weights) {
            std::cout << layer.size() << "\t" << layer[0].size() << std::endl;
        }
    }
    ~NeuralNet();
    NeuralNet(NeuralNet&& other);

    void train(const std::vector<std::vector<float>>& features,
               const std::vector<unsigned long int>& labels);
    std::vector<unsigned long int> predict(const std::vector<std::vector<float>>& features);
private:
    float learning_rate;
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> nodevalues;
    std::vector<std::vector<float>> deltas;
    // no copying
    NeuralNet(const NeuralNet& other) {}
    void forward_propagate(const std::vector<float>& example);
};
} // end namespace nndemo
