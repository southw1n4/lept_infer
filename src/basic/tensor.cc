#include "basic/tensor.h"

#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <string>

#include "basic/tools.h"

namespace leptinfer{

Tensor::Tensor() {
    _init({0}, tensor_type::TYPE_FP32);
}


Tensor::Tensor(std::vector<int> shape, tensor_type type, float val){
    _init(shape, type);
    std::fill_n((float*)data_, size_ / 4, val);
}


Tensor::Tensor(std::vector<int> shape, std::vector<float> vals, tensor_type type){
    _init(shape, type);
    std::copy(vals.begin(), vals.end(), (float*)data_);
}

Tensor::Tensor(const Tensor& rhs){
    _init(rhs.shape_, rhs.type_);
    memcpy(data_, rhs.data_, rhs.size_);
}

Tensor::~Tensor() {
    _free();
}

Tensor& Tensor::operator=(const Tensor& rhs) {
    if(this == &rhs) {
        return *this;
    }
    _free();

    _init(rhs.shape_, rhs.type_);
    memcpy(data_, rhs.data_, rhs.size_);
    return *this;
}

bool Tensor::operator==(const Tensor& rhs) {
    float eps = 1e-4;
    if(rhs.shape_ != shape_) return false;
    float* p1 = (float *)data_;
    float* p2 = (float *)rhs.data_;
    for(int i = 0; i < size_ / 4; ++ i) {
        if(std::abs(p1[i] - p2[i]) > eps) return false;
    }

    return true;
}

float& Tensor::operator()(std::vector<int> idx) {
    int num = size_ / 4, offset = 0;
    for(int i = 0; i < idx.size(); ++ i) {
        num /= shape_[i];
        int p = std::min(shape_[i] - 1, idx[i]);

        offset += p * num;
    }
    return ((float*)data_)[offset];
}

float Tensor::operator()(std::vector<int> idx) const {
    int num = size_ / 4, offset = 0;
    for(int i = 0; i < idx.size(); ++ i) {
        num /= shape_[i];
        int p = std::min(shape_[i] - 1, idx[i]);

        offset += p * num;
    }
    return ((float*)data_)[offset];

}
void Tensor::_init(std::vector<int> shape, tensor_type type) {
    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    switch(type) {
        case ETENOSR_TYPE::TYPE_BOOL:
        case ETENOSR_TYPE::TYPE_CHAR:
            size_ *= 1;
            break;
        case ETENOSR_TYPE::TYPE_SHORT:
            size_ *= 2;
            break;
        case ETENOSR_TYPE::TYPE_FP32:
        case ETENOSR_TYPE::TYPE_INT32:
            size_ *= 4;
            break;
        case ETENOSR_TYPE::TYPE_DOUBLE:
            size_ *= 8;
            break;
        default:
            ERROR("unsupport data type");
    }

    shape_ = shape;
    type_  = type;
    if(size_ > 0) data_ = malloc(size_);
}

void Tensor::_free() {
    if(data_ != NULL && size_ > 0){
        free(data_);
        data_ = NULL;
        size_ = 0;
    }
    shape_.clear();
}

inline
void _output_helper_helper(std::vector<int>& shape, int p, std::vector<int>idx, std::ostream& o){

    int t = idx.back();

}


std::ostream& operator<<(std::ostream& o, const Tensor& t) {

    /*TODO
     *
     */

    return o;
}

}
