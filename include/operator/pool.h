#ifndef __OPERATOR_POOL_H__
#define __OPERATOR_POOL_H__

#include "operator/opbase.h"

namespace leptinfer{

class MaxPool2d: public Op{
 public:
     MaxPool2d(int kernel_size, int stride, int padding);
     ~MaxPool2d(){};

     virtual Tensor operator()(const Tensor&) override;
     virtual void forward() override;
     void compute_local(int tgt_ycoord, int tgt_xcoord, float* tgt_ptr, float* src_ptr, int src_w, int src_h);

 private:
     int kernel_size_;
     int padding_;
     int stride_;

};
}


#endif
