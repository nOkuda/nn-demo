#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "nndemo/Data.h"
#include "nndemo/FileReader.h"
#include "nndemo/NeuralNet.h"

namespace {
std::vector<std::string>& split(const std::string& str,
                           char delim,
                           std::vector<std::string>& elems) {
    std::stringstream ss(str);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> elems;
    split(str, delim, elems);
    return elems;
}

std::tuple<std::unique_ptr<std::vector<std::vector<float>>>,
           std::unique_ptr<std::vector<unsigned long int>>>
parse_file(char* filename) {
    auto features = std::make_unique<std::vector<std::vector<float>>>();
    auto labels = std::make_unique<std::vector<unsigned long int>>();
    nndemo::FileReader ifh(filename);
    std::string strbuf;
    do {
        ifh.getline(strbuf);
        if (strbuf.size() <= 0) {
            // don't try to parse an empty line
            continue;
        }
        std::vector<std::string> elems = split(strbuf, ',');
        features->emplace_back(std::vector<float>());
        for (int i = 0; i < elems.size() - 1; ++i) {
            features->back().push_back(std::stof(elems[i]));
        }
        labels->push_back(std::stoul(elems.back()));
    } while (ifh.good());
    if (!ifh.eof()) {
        std::cerr << "Problem loading file" << std::endl;
        exit(2);
    }
    return std::make_tuple(std::move(features), std::move(labels));
}

template<class URNG>
std::tuple<nndemo::Data, nndemo::NeuralNet> parse_args(int argc, char* argv[], URNG& rng) {
    if (argc != 3) {
        std::cerr << "USAGE: nnapp [data] [architecture]" << std::endl;
        exit(1);
    }
    auto parsed_data = parse_file(argv[1]);
    auto features = std::move(std::get<0>(parsed_data));
    auto labels = std::move(std::get<1>(parsed_data));
    nndemo::FileReader archreader(argv[2]);
    return std::make_tuple(nndemo::Data(features, labels, features->size()),
                           nndemo::NeuralNet(archreader, rng));
}
} // end anonymous namespace

int main(int argc, char* argv[]) {
    std::default_random_engine rng;
    auto parsed = parse_args(argc, argv, rng);
    auto data = std::move(std::get<0>(parsed));
    auto model = std::move(std::get<1>(parsed));
    data.shuffle(rng);
    auto datasplit = data.split_data(0.8);
    auto training_selection = std::get<0>(datasplit);
    auto test_selection = std::get<1>(datasplit);
    auto training = data.get_data(training_selection);
    auto training_features = std::get<0>(training);
    auto training_labels = std::get<1>(training);
    model.train(training_features, training_labels);
    auto test = data.get_data(test_selection);
    auto test_features = std::get<0>(test);
    auto test_labels = std::get<1>(test);
    auto predictions = model.predict(test_features);
    for (int i = 0; i < predictions.size(); i++) {
        std::cout << test_labels[i] << "\t" << predictions[i] << std::endl;
    }
    return 0;
}
