#include "nndemo/Data.h"

#include <utility>

nndemo::Data::Data(std::unique_ptr<std::vector<std::vector<float>>>& features,
                   std::unique_ptr<std::vector<unsigned int>>& labels)
    :
    features(std::move(features)),
    labels(std::move(labels)) {}

nndemo::Data::~Data() {}

nndemo::Data::Data(Data&& other)
    :
    features(std::move(other.features)),
    labels(std::move(other.labels)) {}
