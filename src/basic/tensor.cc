#include "basic/tensor.h"

#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <string>
#include <iostream>

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

std::string _fix(std::string s) {

    bool comma = false;
    for(char c: s) if(c == ',') comma = true;

    int cnt_brackets = 0;
    for(char c: s) if(c == ']') cnt_brackets ++;


    int cnt_endl = 0;
    for(char c: s) if(c == '\n') cnt_endl ++;
    s.clear();

    while(cnt_brackets --) s.push_back(']'); 
    if(comma) s.push_back(',');
    s.push_back('\n'); 

    return s;
}

Tensor Tensor::T() {
    Tensor y({shape_[1], shape_[0]});
    float* p1 = (float *)y.data_;
    float* p2 = (float *)data_; 
    for(int i = 0;  i < shape_[1]; ++ i) {
        for(int j = 0; j < shape_[0]; ++ j) {
            p1[i * shape_[0] + j] = p2[j * shape_[1] + i];
        }
    }
    return y;
}

void Tensor::reshape(const std::vector<int>& shape) {
    int s1 = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    if(s1 != size_ / 4){
        ERROR("cannot reshape between size: %d and size %d\n", s1, (int)size_ / 4);
        exit(0);
    }
    shape_ = shape;
}


void _output_helper(std::vector<int>& shape, int p, std::vector<int>idx, std::ostream& o, const Tensor& t){
    static int bracket_cnt = 0;
    if(idx.empty()) bracket_cnt = shape.size();


    if(p == shape.size() - 1) {
        //output prefix
        {
            for(int i = shape.size() - bracket_cnt; i > 0; i --) o << ' ';
            for(int i = 0; i < bracket_cnt; ++ i) o << '[';

        }

        //output data
        {
            for(int i = 0; i < shape[p]; ++ i) {
                idx.push_back(i);
                o << t(idx) ;
                if(i != shape[p] - 1) o << ", ";
                idx.pop_back();
            }
        }

        //output end 
        {
            int cnt = 0;
            for(int i = shape.size() - 1; i >= 0 ; -- i) {
                if(idx[i] != shape[i] - 1) break;
                cnt ++;
            }
            bracket_cnt = cnt;
            while(cnt --)o <<']';
            if(bracket_cnt != shape.size()){
                o << ",";
                for(int i = 0; i < bracket_cnt; ++ i) o << '\n';
            }
        }


        return ;
    }

    for(int i = 0; i < shape[p]; ++ i) {

        idx.push_back(i);
        _output_helper(shape, p + 1, idx, o, t);
        idx.pop_back();
    }

}


std::ostream& operator<<(std::ostream& o, const Tensor& t) {

    std::vector<int> idx;
    auto shape = t.shape();

    _output_helper(shape, 0, idx, o, t);

    return o;
}

std::ostream& operator<<(std::ostream& o, std::vector<int>& s){

    o << '[';
    for(int i = 0; i < s.size(); ++ i) {
        if(i == s.size() - 1) o << s[i] << ']';
        else o << s[i] << ", ";
    }
    return o;

}

}
