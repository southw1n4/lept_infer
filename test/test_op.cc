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
            printf("%s\t......................................[OK]\n", #func); \
        }\
        else{\
            printf("%s\t..................................[FAILED]\n", #func); \
        }\
    }while(0)

bool test_linear(){
    auto op = leptinfer::Linear(2, 3);
    auto x = Tensor({3, 2}, {1, 2, 3, 4, 5, 6});
    op.set_weight(Tensor({2, 3}, {6, 5, 4, 3 ,2, 1}));
    op.set_bias(Tensor({1, 3}, {0.4, 0.5, 0.6}));

    return op(x) == Tensor({3, 3}, {12.4, 9.5, 6.6,
                                    30.4, 23.5, 16.6,
                                    48.4, 37.5, 26.6});
}

void test_op() {
    printf("---------------------------TEST_OP------------------------\n");
    TEST(test_linear);

    printf("TEST_OP: %d/%d\n",op_true, op_cnt);
}


