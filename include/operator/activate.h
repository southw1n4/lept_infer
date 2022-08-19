#ifndef __OPERATOR_SOFTMAT_H__
#define __OPERATOR_SOFTMAT_H__

#include "operator/opbase.h"

namespace leptinfer{

class Softmax: public Op{
 public:
     Softmax(int dim = 0):dim_(dim){all_cnt = now_cnt = 1;}
     virtual ~Softmax() override {};
     virtual Tensor operator()(const Tensor&) override;
     virtual void forward() override;
 private:
     int dim_ = 0;
};

class TanH: public Op{
 public:
     TanH(){all_cnt = now_cnt = 1;}
     virtual ~TanH() override {};
     virtual Tensor operator()(const Tensor&) override;
     virtual void forward() override;
};

class ReLU: public Op{
 public:
     ReLU(){all_cnt = now_cnt = 1;}
     ~ReLU(){};

     virtual Tensor operator()(const Tensor&) override;
     virtual void forward() override;


};

}

#endif
