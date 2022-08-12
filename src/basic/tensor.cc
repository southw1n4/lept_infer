#include "basic/tensor.h"

#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>

#include "basic/tools.h"

namespace leptinfer{


Tensor::Tensor(std::vector<int> shape, tensor_type type, float val){
    _init(shape, type);
    std::fill_n((float*)data_, size_, val);
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
    if(rhs.shape_ != shape_) return false;
    return memcmp(data_, rhs.data_, size_) == 0;
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
}

void Tensor::_free() {
    if(data_ != NULL && size_ > 0){
        free(data_);
        data_ = NULL;
        size_ = 0;
    }
    shape_.clear();
}

}
