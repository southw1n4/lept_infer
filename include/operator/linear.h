#ifndef __BASIC_LINEAR_H__
#define __BASIC_LINEAR_H__

#include "operator/opbase.h"


namespace leptinfer{
class Linear: public Op{
 public:
    Linear(int input_dim, int output_dim, bool required_bias = true);
    ~Linear();
    virtual Tensor operator()(const Tensor&) override;

 public:
    void set_weight(const Tensor&);
    void set_bias(const Tensor&);

 private:
    Tensor weight_;
    Tensor bias_;
    bool required_bias_ = true;


}; 
}

#endif
