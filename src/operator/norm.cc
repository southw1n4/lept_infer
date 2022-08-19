#include "operator/norm.h"

#include "basic/tools.h"
#include <iostream>

namespace leptinfer{
BatchNorm2d::BatchNorm2d(int num_features, float eps):
    num_features_(num_features), eps_(eps){now_cnt = all_cnt = 1;}

BatchNorm2d::~BatchNorm2d(){
    if(beta_ != NULL){
        delete beta_;
        beta_ = NULL;
    }

    if(gamma_ != NULL) {
        delete gamma_;
        gamma_ = NULL;
    }
}

void BatchNorm2d::forward() {

    auto y = (*this)(*in[0]);
    notify(std::make_shared<Tensor>(y));
}

void BatchNorm2d::set_gamma(const Tensor& a) {
    if(gamma_ != NULL) delete gamma_;
    gamma_ = new Tensor(a);
}

void BatchNorm2d::set_beta(const Tensor& a) {
    if(beta_ != NULL) delete beta_;
    beta_ = new Tensor(a);
}

Tensor BatchNorm2d::operator()(const Tensor& x) {
    Tensor y(x);
    
    if(x.shape().size() != 4) {
        ERROR("wrong shape");
    }

#ifdef HAND

    auto shape = x.shape();
    const int B = shape[0];
    const int C = shape[1];
    const int H = shape[2];
    const int W = shape[3];

    auto EPS = Tensor({1, C, 1, 1}, Tensor::tensor_type::TYPE_FP32, eps_);
    auto channel_sum = sum(sum(sum(x, 2), 3), 0);

    auto channel_mean = div(channel_sum, Tensor({1, C, 1, 1}, Tensor::tensor_type::TYPE_FP32, B * H * W));
    auto x_demean_pow = mut(sub(x, channel_mean), sub(x, channel_mean));
    auto channel_var = div(sum(sum(sum(x_demean_pow, 2), 3), 0), Tensor({1, C, 1, 1}, Tensor::tensor_type::TYPE_FP32, B * H * W - 1));

    y = div(sub(y, channel_mean), add(sqrt(channel_var), EPS));
    y = add(mut(*beta_, y), *gamma_);

#endif
    return y;

}

}
