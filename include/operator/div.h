#ifndef __OPERATOR_DIV_H__
#define __OPERATOR_DIV_H__

#include "operator/opbase.h"

namespace leptinfer{

class Div: public Op{
 public:
    Div(int cnt){all_cnt = now_cnt = cnt;}
    ~Div();

    virtual Tensor operator()(const Tensor& a) override;
    virtual void forward() override;
    void set_constant(Tensor* a){constant_ = a;}
 private:
    Tensor*  constant_ = NULL;

};
}


#endif


