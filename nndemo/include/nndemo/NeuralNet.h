#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <string>

#include "nndemo/FileReader.h"

namespace nndemo {
class NeuralNet {
public:
    template<class URNG> NeuralNet(FileReader& ifh, URNG& rng) {
        std::vector<std::string> parsed;
        std::string strbuf;
        do {
            ifh.getline(strbuf);
            if (strbuf.size() == 0) {
                continue;
            }
            parsed.push_back(strbuf);
        } while (ifh.good());
        if (!ifh.eof()) {
            std::cerr << "Network architecture parsing problem" << std::endl;
        }
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        learning_rate = stof(parsed.front());
        for (int i = 2; i < parsed.size()-1; ++i) {
            weights.emplace_back(std::vector<std::vector<float>>());
            // account for bias
            int weightscount = stoi(parsed[i-1]) + 1;
            nodevalues.emplace_back(std::vector<float>(weightscount));
            *(nodevalues.back().rbegin()) = 1.0;
            int nodescount = stoi(parsed[i]);
            for (int j = 0; j < nodescount; ++j) {
                weights.back().push_back(std::vector<float>());
                for (int k = 0; k < weightscount; k++) {
                    // initialize weights randomly
                    weights.back().back().push_back((distribution(rng) - 0.5) * 0.001);
                }
            }
        }
        // output layer doesn't have weights out from it
        nodevalues.emplace_back(std::vector<float>(stoi(parsed.back())+1));
        *(nodevalues.back().rbegin()) = 1.0;
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
    // no copying
    NeuralNet(const NeuralNet& other) {}
    void forward_propagate(const std::vector<float>& example);
};
} // end namespace nndemo
