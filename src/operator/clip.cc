#include "operator/clip.h"

namespace leptinfer{

 
Tensor Clip::operator()(const Tensor& a){
    auto y = a;
    float low = ((float *)low_->data())[0];
    float up = ((float *)up_->data())[0];
    float* data = (float *)y.data();

    for(int i = 0; i < y.size(); ++ i){
        data[i] = std::min(std::max(data[i], low), up);
    }
    return y;
}

void Clip::forward(){
    auto y = (*this)(*in[0]);
    notify(std::make_shared<Tensor>(y));
}

Clip::~Clip(){
    if(low_!= NULL)
        delete low_;
    if(up_ != NULL){
        delete up_;
    }
}

}
