#ifndef __OPERATOR_OP_BASE_H__
#define __OPERATOR_OP_BASE_H__

#include <vector>
#include <unordered_map>
#include <memory>

#include "basic/tensor.h"

namespace leptinfer{
class Op{
 public:
    virtual ~Op();
    virtual Tensor operator()(const Tensor& x) = 0;
    virtual void forward() = 0;

    void notify(std::shared_ptr<Tensor>);
    void wakeup();
    static void run(Op*);

    std::string op_name_;
    std::vector<Op *> next_op;
    std::unordered_map<std::string, bool> output;
    std::vector<std::shared_ptr<Tensor>> in;
    int now_cnt = 0;
    int all_cnt = 0;
    bool status = false;
    bool is_output = false;
    std::shared_ptr<Tensor> result = NULL;

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
