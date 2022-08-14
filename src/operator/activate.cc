#include "operator/activate.h"

#include <stdio.h>


namespace leptinfer{

Tensor Softmax::operator()(const Tensor& x) {
    return div(exp(x), sum(exp(x), dim_));
}

Tensor TanH::operator()(const Tensor& x){
    return tanh(x);
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

}

