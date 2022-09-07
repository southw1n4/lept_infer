#ifndef __OPERATOR_ADD_H__
#define __OPERATOR_ADD_H__

#include "operator/opbase.h"

namespace leptinfer{

class Add: public Op{
 public:
    Add(int cnt){all_cnt = now_cnt = cnt;}
    ~Add();

    virtual Tensor operator()(const Tensor& a) override;
    virtual void forward() override;
    void set_constant(Tensor* a){constant_ = a;}
 private:
    Tensor*  constant_ = NULL;

};
}


#endif

