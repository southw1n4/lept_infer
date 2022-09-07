#include "operator/div.h"

namespace leptinfer{

 
Tensor Div::operator()(const Tensor& a){
    if(constant_->shape().size() == 1){
        constant_->reshape(std::vector<int>(a.shape().size(), 1));
    }

    return div(a, *constant_);
}

void Div::forward(){
    if(constant_ == NULL){
        constant_ = new Tensor(*in[0]);
    }
    auto y = (*this)(*in[1]);
    notify(std::make_shared<Tensor>(y));
}

Div::~Div(){
    if(constant_ != NULL)
        delete constant_;
}

}




