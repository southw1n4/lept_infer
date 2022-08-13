#ifndef __OPERATOR_CONV_H__
#define __OPERATOR_CONV_H__

#include "operator/opbase.h"

namespace leptinfer{
class Conv: public Op{
 public:
     Conv(int input_dims, int output_dims, int kernel_size, int padding, int stride);
     ~Conv();
     virtual Tensor operator()(const Tensor&);

 public:
     void set_kernel(const Tensor&);

 private:
     Tensor* kernel_;
     int input_dims_;
     int output_dims_;
     int kernel_size_;
     int padding_;
     int stride_;
};

}

#endif
