#ifndef __NET_NETBASE_H__
#define __NET_NETBASE_H__

#include "basic/tensor.h"
#include "operator/opbase.h"


namespace leptinfer{

class Net{
 public:
    virtual ~Net();
    virtual std::vector<Tensor> excute(const Tensor&);

    std::vector<Op*> ops; 
    std::vector<int> output_idx;
    std::vector<Tensor> output;
};
}


#endif





