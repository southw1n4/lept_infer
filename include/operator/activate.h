#ifndef __OPERATOR_SOFTMAT_H__
#define __OPERATOR_SOFTMAT_H__

#include "operator/opbase.h"

namespace leptinfer{

class Softmax: public Op{
 public:
     Softmax(int dim = 0):dim_(dim){}
     virtual ~Softmax() override {};
     virtual Tensor operator()(const Tensor&) override;
 private:
     int dim_ = 0;
};

class TanH: public Op{
 public:
     virtual ~TanH() override {};
     virtual Tensor operator()(const Tensor&) override;
};

class ReLU: public Op{
 public:
     ReLU(){};
     ~ReLU(){};

     virtual Tensor operator()(const Tensor&) override;


};

}

#endif
