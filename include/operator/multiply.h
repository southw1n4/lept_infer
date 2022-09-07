#ifndef __OPERATOR_MULTIPLY_H__
#define __OPERATOR_MULTIPLY_H__

#include "operator/opbase.h"

namespace leptinfer{

class Mul: public Op{
 public:
    Mul(int cnt){all_cnt = now_cnt = cnt;}
    ~Mul();

    virtual Tensor operator()(const Tensor& a) override;
    virtual void forward() override;
    void set_constant(Tensor* a){constant_ = a;}
 private:
    Tensor*  constant_ = NULL;

};
}


#endif


