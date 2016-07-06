#pragma once
#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

namespace nndemo {
class Data {
public:
    Data(std::unique_ptr<std::vector<std::vector<float>>>& features,
         std::unique_ptr<std::vector<unsigned long int>>& labels,
         unsigned long int datacount);
    ~Data();
    // only moving
    Data(Data&& other);

    template<class URNG> void shuffle(URNG& g) {
        std::shuffle(std::begin(order), std::end(order), g);
    }

    std::tuple<std::vector<std::vector<float>>, std::vector<unsigned long int>>
    get_data(const std::vector<unsigned long int>& selection);

    std::tuple<std::vector<unsigned long int>, std::vector<unsigned long int>>
    split_data(float proportion);

    void print_order() const;
    int size() const;
    int features_size() const;
private:
    std::unique_ptr<std::vector<std::vector<float>>> features;
    std::unique_ptr<std::vector<unsigned long int>> labels;
    std::vector<unsigned long int> order;
    // no copying
    Data(const Data& other) {}
};
} // end namespace nndemo
