#pragma once
#include <memory>
#include <vector>

#include "nndemo/FileReader.h"

namespace nndemo {
class Data {
public:
    Data(std::unique_ptr<std::vector<std::vector<float>>>& features,
         std::unique_ptr<std::vector<unsigned int>>& labels);
    ~Data();
    // only moving
    Data(Data&& other);
private:
    std::unique_ptr<std::vector<std::vector<float>>> features;
    std::unique_ptr<std::vector<unsigned int>> labels;
    // no copying
    Data(const Data& other) {}
};
} // end namespace nndemo
