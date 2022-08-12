#include "operator/linear.h"

#include "basic/tensor.h"
#include "basic/tools.h"


namespace leptinfer{

Linear::Linear(int input_dim, int output_dim, bool required_bias) {
    required_bias_ = required_bias;
}

Linear::~Linear() {
}

void Linear::set_weight(const Tensor& rhs) {
    weight_ = rhs;
}

void Linear::set_bias(const Tensor& rhs) {
    required_bias_ = true;
    bias_ = rhs;
}


Tensor Linear::operator()(const Tensor& x) {
    Tensor y;
    if(x.shape().size() != 2 || x.shape().back() != weight_.shape()[0]) {
        ERROR("dimension mismatch");
    }
    auto shape1 = x.shape(), shap2 = weight_.shape();
    int na = shape1[0], ma = shape1[1];
    int nb = shap2[0], mb = shap2[1];
    y = Tensor({na, mb});

#ifdef HAND

    float* pw = (float *) weight_.data(); 
    float* px = (float *) x.data();
    float* py = (float *) y.data();
    for(int i = 0; i < na; ++ i) {
        for(int j = 0; j < mb; ++ j) {
            if(required_bias_) {
                *(py + i * mb + j) = ((float *)bias_.data())[j];
            }
            for(int k = 0; k < ma; ++ k) {
                *(py + i * mb + j) += *(pw + i * ma + k) + *(py + k * na + j) ;
            }

        }
    }

#endif

    return y;
}

}
