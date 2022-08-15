#ifndef __NET_NETBASE_H__
#define __NET_NETBASE_H__

#include "basic/tensor.h"
#include "operator/opbase.h"


namespace leptinfer{

class Net{
 public:
    virtual ~Net(){};
    virtual Tensor operator()(const Tensor&) = 0;
    static void top_sort(std::vector<Op*>&);

    std::vector<Op*> ops; 
};
}


#endif





