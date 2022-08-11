#ifndef __BASIC_TENSOR_H__
#define __BASIC_TENSOR_H__

#include <cstddef>
#include <vector>

namespace leptinfer{

class Tensor{
 public:
     enum ETENOSR_TYPE{
         TYPE_BOOL = 0,
         TYPE_INT8,
         TYPE_CHAR,
         TYPE_FP16,
         TYPE_SHORT,
         TYPE_FP32,
         TYPE_INT32,
         TYPE_DOUBLE,
         TYPE_UNKNOWN
     };

     using tensor_type = ETENOSR_TYPE;
     Tensor() = delete;

     Tensor(std::vector<int> shape, tensor_type type = ETENOSR_TYPE::TYPE_FP32, float val = 0);
     Tensor(std::vector<int> shape, std::vector<bool> vals, tensor_type type = ETENOSR_TYPE::TYPE_BOOL);
     Tensor(std::vector<int> shape, std::vector<char> vals, tensor_type type = ETENOSR_TYPE::TYPE_CHAR);
     Tensor(std::vector<int> shape, std::vector<float> vals, tensor_type type = ETENOSR_TYPE::TYPE_FP32);
     Tensor(std::vector<int> shape, std::vector<int> vals, tensor_type type = ETENOSR_TYPE::TYPE_INT32);
     Tensor(std::vector<int> shape, std::vector<double> vals, tensor_type type = ETENOSR_TYPE::TYPE_DOUBLE);
     Tensor(const Tensor& rhs);

     ~Tensor();

 public:
     std::vector<int> shape() const {return shape_;}
     int shape(int idx) const {return shape_[idx];}
     const void* data() const {return data_;}
     void* data() {return data_;}
     tensor_type type() const {return type_;}

public:
     Tensor& operator=(const Tensor& rhs);


 private:
     void _init(std::vector<int> shape, tensor_type type);
     void _free();

 private:
     size_t size_;
     tensor_type type_;
     std::vector<int> shape_;
     void* data_;
};

}

#endif
