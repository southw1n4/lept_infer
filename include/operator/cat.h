#ifndef __OPERATOR_CAT_H__
#define __OPERATOR_CAT_H__

#include "operator/opbase.h"

namespace leptinfer{

class Cat: public Op{
 public:
    Cat(int cnt, int dim){dim_ = dim; all_cnt = now_cnt = cnt;}
    ~Cat(){};

    virtual Tensor operator()(const Tensor& a) override;
    virtual void forward() override;
 private:
    int dim_;

};
}


#endif


