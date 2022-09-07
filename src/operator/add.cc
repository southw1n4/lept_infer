#include "operator/add.h"

namespace leptinfer{

 
Tensor Add::operator()(const Tensor& a){
    if(constant_->shape().size() == 1){
        constant_->reshape(std::vector<int>(a.shape().size(), 1));
    }

    return add(a, *constant_);
}

void Add::forward(){
    if(constant_ == NULL){
        constant_ = new Tensor(*in[0]);
    }
    auto y = (*this)(*in[1]);
    notify(std::make_shared<Tensor>(y));
}

Add::~Add(){
    if(constant_ != NULL)
        delete constant_;
}

}



