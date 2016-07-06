#include "nndemo/Data.h"

#include <iostream>
#include <numeric>
#include <utility>

nndemo::Data::Data(std::unique_ptr<std::vector<std::vector<float>>>& features,
                   std::unique_ptr<std::vector<unsigned long int>>& labels,
                   unsigned long int datacount)
        : features(std::move(features)),
          labels(std::move(labels)),
          order(datacount) {
    std::iota(order.begin(), order.end(), 0);
}

nndemo::Data::~Data() {}

nndemo::Data::Data(Data&& other)
        : features(std::move(other.features)),
          labels(std::move(other.labels)),
          order(other.order) {}

std::tuple<std::vector<std::vector<float>>, std::vector<unsigned long int>>
nndemo::Data::get_data(const std::vector<unsigned long int>& selection) {
    std::vector<std::vector<float>> out_features;
    std::vector<unsigned long int> out_labels;
    for (auto curnum : selection) {
        out_features.emplace_back(std::vector<float>(features->at(curnum)));
        out_labels.push_back(labels->at(curnum));
    }
    return std::make_tuple(out_features, out_labels);
}

std::tuple<std::vector<unsigned long int>, std::vector<unsigned long int>>
nndemo::Data::split_data(float proportion) {
    int threshold = order.size() * proportion;
    std::vector<unsigned long int> training_selection;
    std::vector<unsigned long int> test_selection;
    return std::make_tuple(std::vector<unsigned long int>(std::begin(order), std::begin(order)+threshold),
                           std::vector<unsigned long int>(std::begin(order)+threshold, std::end(order)));
}

void nndemo::Data::print_order() const {
    for (auto& ord : order) {
        std::cout << ord << std::endl;
    }
}

int nndemo::Data::size() const {
    return features->size();
}

int nndemo::Data::features_size() const {
    return features->front().size();
}
