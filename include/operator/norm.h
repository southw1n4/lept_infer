#ifndef __OPERATOR_NORM_H__
#define __OPERATOR_NORM_H__

#include "operator/opbase.h"

namespace leptinfer{

class BatchNorm2d: public Op{
 public:
    BatchNorm2d(int num_features, float eps = 1e-6);
    ~BatchNorm2d();

    virtual Tensor operator()(const Tensor& a) override;
    virtual void forward() override;

    void set_gamma(Tensor*);
    void set_beta(Tensor*);
    void set_running_mean(Tensor*);
    void set_running_var(Tensor*);

 private:
    int num_features_;
    float eps_;
    Tensor* gamma_ = NULL;
    Tensor* beta_ = NULL;
    Tensor* running_mean_ = NULL;
    Tensor* running_var_ = NULL;

};
}


#endif
