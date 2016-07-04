#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "nndemo/Data.h"
#include "nndemo/FileReader.h"
#include "nndemo/NeuralNet.h"

namespace {
std::tuple<std::unique_ptr<std::vector<std::vector<float>>>,
           std::unique_ptr<std::vector<unsigned int>>>
parse_file(char* filename) {
    // TODO parse iris file
    return std::make_tuple(std::make_unique<std::vector<std::vector<float>>>(),
                           std::make_unique<std::vector<unsigned int>>());
}

std::tuple<nndemo::Data, nndemo::NeuralNet> parse_args(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "USAGE: nnapp [data] [architecture]" << std::endl;
        exit(1);
    }
    auto parsed_data = parse_file(argv[1]);
    auto features = std::move(std::get<0>(parsed_data));
    auto labels = std::move(std::get<1>(parsed_data));
    return std::make_tuple(nndemo::Data(features, labels), nndemo::NeuralNet());
}
} // end anonymous namespace

int main(int argc, char* argv[]) {
    auto parsed = parse_args(argc, argv);
    auto data = std::move(std::get<0>(parsed));
    auto model = std::move(std::get<1>(parsed));
    std::cout << "Initial" << std::endl;
    nndemo::FileReader ifh(argv[1]);
    std::string strbuf;
    do {
        ifh.getline(strbuf);
        std::cout << strbuf << std::endl;
    } while (ifh.good());
    return 0;
}
