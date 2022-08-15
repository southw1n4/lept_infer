#ifndef __OPERATOR_NORM_H__
#define __OPERATOR_NORM_H__

#include "operator/opbase.h"

namespace leptinfer{

class BatchNorm2d: public Op{
 public:
    BatchNorm2d(int num_features, float eps = 1e-6);
    ~BatchNorm2d();

    virtual Tensor operator()(const Tensor& a) override;

    void set_gamma(const Tensor&);
    void set_beta(const Tensor&);

 private:
    int num_features_;
    float eps_;
    Tensor* gamma_ = NULL;
    Tensor* beta_ = NULL;

};
}


#endif
