#ifndef __OPERATOR_CLIP_H__
#define __OPERATOR_CLIP_H__

#include "operator/opbase.h"

namespace leptinfer{

class Clip: public Op{
 public:
    Clip(int cnt){all_cnt = now_cnt = cnt;}
    ~Clip();

    virtual Tensor operator()(const Tensor& a) override;
    virtual void forward() override;
    void set_low(Tensor* a){low_ = a;}
    void set_up(Tensor* a){up_ = a;}
 private:
    Tensor*  low_ = NULL;
    Tensor*  up_ = NULL;

};
}


#endif


