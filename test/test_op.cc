#include <cstdio>
#include "operator/linear.h"

static int op_cnt = 0;
static int op_true = 0;

using namespace leptinfer;
#define TEST(func) \
    do{\
        op_cnt ++;\
        if(func()) {\
            op_true++;\
            printf("%s\t.........................[OK]", #func); \
        }\
        else{\
            printf("%s\t......................[FAILED]", #func); \
        }\
    }while(0)

bool test_linear(){
    auto op = leptinfer::Linear(2, 3);
    auto x = Tensor({3, 2}, {1, 1, 1, 1, 1, 1});
    op.set_weight(Tensor({2, 3}, {1, 1, 1, 1 ,1, 1}));
    op.set_bias(Tensor({3}, {0.5, 0.5, 0.5}));

    return op(x) == Tensor({3, 3}, std::vector<float>(9, 2.5));
}

void test_op() {
    TEST(test_linear);

    printf("TEST_OP: %d/%d\n",op_true, op_cnt);
}


