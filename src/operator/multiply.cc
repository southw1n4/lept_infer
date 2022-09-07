#include "operator/multiply.h"

namespace leptinfer{

 
Tensor Mul::operator()(const Tensor& a){
    if(constant_->shape().size() == 1){
        constant_->reshape(std::vector<int>(a.shape().size(), 1));
    }

    return mut(a, *constant_);
}

void Mul::forward(){
    if(constant_ == NULL){
        constant_ = new Tensor(*in[0]);
    }
    auto y = (*this)(*in[1]);
    notify(std::make_shared<Tensor>(y));
}

Mul::~Mul(){
    if(constant_ != NULL)
        delete constant_;
}

}




