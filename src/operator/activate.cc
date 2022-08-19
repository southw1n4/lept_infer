#include "operator/activate.h"

#include <stdio.h>


namespace leptinfer{

Tensor Softmax::operator()(const Tensor& x) {
    return div(exp(x), sum(exp(x), dim_));
}

void Softmax::forward() {

    auto y = (*this)(*in[0]);
    in[0] = NULL;
    notify(std::make_shared<Tensor>(y));
}

Tensor TanH::operator()(const Tensor& x){
    return tanh(x);
}

void TanH::forward() {

    auto y = (*this)(*in[0]);
    in[0] = NULL;
    notify(std::make_shared<Tensor>(y));
}

Tensor ReLU::operator()(const Tensor& x){

    Tensor y(x);

#ifdef HAND
    int size = y.size();
    float* data = (float *)y.data();
    for(int i = 0; i < size; ++ i){ 
        data[i] = std::max(0.f, data[i]);
    }
#endif

    return y;
}

void ReLU::forward() {

    auto y = (*this)(*in[0]);
    in[0] = NULL;
    notify(std::make_shared<Tensor>(y));
}

}

