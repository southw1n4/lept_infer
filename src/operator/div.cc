#include "operator/div.h"
#include <iostream>

namespace leptinfer{

 
Tensor Div::operator()(const Tensor& a){
    int dims = constant_->shape().size();
    if(dims == 1){
        auto new_shape = std::vector<int>(a.shape().size(), 1);
        constant_->reshape(new_shape);
    }
    return div(a, *constant_);
}

void Div::forward(){
    Tensor y;
    if(constant_ == NULL){
        constant_ = new Tensor(*in[0]);
        y = (*this)(*in[1]);
    }else y = (*this)(*in[0]);

    notify(std::make_shared<Tensor>(y));
}

Div::~Div(){
    if(constant_ != NULL)
        delete constant_;
}

}




