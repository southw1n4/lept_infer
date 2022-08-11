#ifndef __BASICE_TOOLS_H__
#define __BASICE_TOOLS_H__


#define LOG_BASE(type, ...)\
    do{ \
        printf("[%s] [%s:%d]: ", #type, __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
    }while(0)


#define INFO(...) LOG_BASE(I,  __VA_ARGS__)
#define WRANING(...) LOG_BASE(W, __VA_ARGS__)
#define ERROR(...) LOG_BASE(E, __VA_ARGS__)

#define TYPE_TO_PTR(ptr, type) \
    ({\
        switch(type){\
            case ETENOSR_TYPE::TYPE_BOOL: \
                ptr = (bool *)ptr; break;\
            case ETENOSR_TYPE::TYPE_CHAR: \
                ptr = (char *)ptr; break;\
            case ETENOSR_TYPE::TYPE_SHORT: \
                ptr = (short *)ptr; break;\
            case ETENOSR_TYPE::TYPE_FP32: \
                ptr = (float *)ptr; break;\
            case ETENOSR_TYPE::TYPE_INT32: \
                ptr = (int *) ptr; break;\
            case ETENOSR_TYPE::TYPE_DOUBLE: \
                ptr = (double *) ptr; break;\
            default: \
                ERROR("unsupport data type"); \
        }\
        ptr;\
    })

#endif
