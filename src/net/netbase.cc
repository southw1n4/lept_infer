
#include "net/netbase.h"
#include <thread>


namespace leptinfer{

Net::~Net() {

    for(auto op: ops) {
        delete op;
    }
}
std::vector<Tensor> Net::inference(const Tensor& a){

    ops[0]->in.push_back(std::make_shared<Tensor>(a));
    ops[0]->forward();

    while(true) {
        bool flag = true;
        for(int i = 0; i < ops.size(); ++ i){ 
            if(!ops[i]->status){
                flag = false;
                break;
            }
        }
        if(flag) break;
    }

    for(int i = 0; i < ops.size(); ++ i) {
        ops[i]->status = false;
        if(ops[i]->is_output) {
            output.push_back(*ops[i]->result);
            ops[i]->result = NULL;
        }
    }; 


    return output;
}


}
