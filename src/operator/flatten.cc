#include "operator/flatten.h"

#include <numeric>
#include <functional>
#include <iostream>

namespace leptinfer{

Tensor Flatten::operator()(const Tensor& x) {
    std::vector<int> shape = x.shape();
    Tensor y(x);
    y.reshape({shape[0], std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<int>())});
    return y;
}

void Flatten::forward() {

    auto y = (*this)(*in[0]);
    notify(std::make_shared<Tensor>(y));
}

}
