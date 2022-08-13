#include "operator/activate.h"


namespace leptinfer{

Tensor Softmax::operator()(const Tensor& x) {
    return div(exp(x), sum(exp(x), dim_));
}

Tensor TanH::operator()(const Tensor& x){
    return tanh(x);
}

}

