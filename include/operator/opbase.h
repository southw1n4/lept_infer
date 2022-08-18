#ifndef __OPERATOR_OP_BASE_H__
#define __OPERATOR_OP_BASE_H__

#include <vector>
#include <unordered_map>

#include "basic/tensor.h"

namespace leptinfer{
class Op{
 public:
    virtual ~Op(){};
    virtual Tensor operator()(const Tensor&) = 0;

    std::vector<Op *> next;
    std::unordered_map<std::string, bool> output;

};

Tensor gemm(const Tensor&, const Tensor&);

Tensor add(const Tensor&, const Tensor&);

Tensor sum(const Tensor&, int dim = -1);
Tensor sub(const Tensor&, const Tensor&);
Tensor mut(const Tensor&, const Tensor&);
Tensor div(const Tensor&, const Tensor&);

Tensor exp(const Tensor&);
Tensor sqrt(const Tensor&);


Tensor tanh(const Tensor&);

}

#endif
