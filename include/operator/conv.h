#ifndef __OPERATOR_CONV_H__
#define __OPERATOR_CONV_H__

#include "operator/opbase.h"

namespace leptinfer{
class Conv2d: public Op{
 public:
     Conv2d(int input_dims, int output_dims, int kernel_size, int stride, int padding, bool required_bias = false);
     ~Conv2d();
     virtual Tensor operator()(const Tensor&);

 public:
     void set_weight(const Tensor&);
     void set_bias(const Tensor&);
 

 private: 
     void compute_local(int tgt_xcoord, int tgt_ycoord, float* src_ptr, float* tgt_ptr, float* weight_ptr, int src_offset, int weight_offset, int src_w, int src_h, float bias = 0.f);


 private:
     Tensor* weight_ = NULL;
     Tensor* bias_ = NULL;
     int input_dims_;
     int output_dims_;
     int kernel_size_;
     int padding_;
     int stride_;
     bool required_bias_ = false;
};

}

#endif