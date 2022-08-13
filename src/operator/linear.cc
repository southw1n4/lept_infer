#include "operator/linear.h"

#include "basic/tensor.h"
#include "basic/tools.h"


namespace leptinfer{

Linear::Linear(int input_dim, int output_dim, bool required_bias) {
    required_bias_ = required_bias;
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

void Linear::set_weight(const Tensor& rhs) {
    weight_ = new Tensor(rhs);
}

void Linear::set_bias(const Tensor& rhs) {
    required_bias_ = true;
    bias_ = new Tensor(rhs);
}


Tensor Linear::operator()(const Tensor& x) {

    return add(gemm(x, *weight_), *bias_);
}

}
