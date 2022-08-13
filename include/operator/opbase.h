#ifndef __OPERATOR_OP_BASE_H__
#define __OPERATOR_OP_BASE_H__

#include "basic/tensor.h"

namespace leptinfer{
class Op{
 public:
    virtual ~Op(){};
    virtual Tensor operator()(const Tensor&) = 0;

};

Tensor gemm(const Tensor&, const Tensor&);

Tensor add(const Tensor&, const Tensor&);

Tensor sum(const Tensor&, int dim = -1);
Tensor sub(const Tensor&, const Tensor&);
Tensor mut(const Tensor&, const Tensor&);
Tensor div(const Tensor&, const Tensor&);

Tensor exp(const Tensor&);


Tensor tanh(const Tensor&);

}

#endif
