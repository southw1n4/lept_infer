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

void BatchNorm2d::set_gamma(Tensor* a) {
    if(gamma_ != NULL) delete gamma_;
    gamma_ = a;
}

void BatchNorm2d::set_beta(Tensor* a) {
    if(beta_ != NULL) delete beta_;
    beta_ = a;
}
void BatchNorm2d::set_running_mean(Tensor* a) {
    if(running_mean_ != NULL) delete running_mean_;
    running_mean_ = a;
}


void BatchNorm2d::set_running_var(Tensor* a) {
    if(running_var_ != NULL) delete running_var_;
    running_var_ = a;
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
    y = sub(x, *running_mean_);
    y = div(y, sqrt(add(*running_var_, EPS)));
    y = add(mut(y, *beta_), *gamma_);


#endif
    return y;

}

}
