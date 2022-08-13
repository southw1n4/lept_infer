#include "operator/conv.h"


namespace leptinfer{
Conv::Conv(int input_dims, int output_dims, int kernel_size, int padding, int stride):
    input_dims_(input_dims), 
    output_dims_(output_dims), 
    kernel_size_(kernel_size),
    padding_(padding),
    stride_(stride){}



Tensor Conv::operator()(const Tensor& a){

#ifdef HADN 

    /*TODO
     *
     */

#endif

    return a;
}
}
