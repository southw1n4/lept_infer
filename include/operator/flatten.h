#ifndef __OPERATOR_FLATTEN_H__
#define __OPERATOR_FLATTEN_H__

#include "operator/opbase.h"

namespace leptinfer{
class Flatten: public Op{
 public:
     Flatten(){all_cnt = now_cnt = 1;}
     virtual Tensor operator()(const Tensor&) override;
     virtual void forward() override;

};

}


#endif
