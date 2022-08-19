#include "operator/linear.h"

#include "basic/tensor.h"
#include "basic/tools.h"
#include <iostream>


namespace leptinfer{

Linear::Linear(int input_dim, int output_dim, bool required_bias) {
    required_bias_ = required_bias;
    now_cnt = all_cnt = 1;
}

Linear::~Linear() {
    if(weight_ != NULL) {
        delete weight_;
        weight_ = NULL;
    }

    if(bias_ != NULL) {
        delete bias_;
        bias_ = NULL;
    }
}

void Linear::set_weight(Tensor* rhs) {
    if(weight_ != NULL) delete weight_; 
    weight_ = rhs;
}

void Linear::set_bias(Tensor* rhs) {
    required_bias_ = true;
    if(bias_ != NULL) delete bias_;
    bias_ = rhs;
}


Tensor Linear::operator()(const Tensor& x) {
    return add(gemm(x, *weight_), *bias_);
}

void Linear::forward() {

    auto y = (*this)(*in[0]);
    notify(std::make_shared<Tensor>(y));
}

}
