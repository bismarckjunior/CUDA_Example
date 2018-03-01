/*******************************************************************************
| Project: CUDA Example                                                        |
| Author:  Bismarck Gomes Souza Junior <bismarck@puc-rio.br>                   |
\******************************************************************************/
#ifndef _CEXAMPLE_H_
#define _CEXAMPLE_H_

#include "CudaVector.h"

// Launch kernels
extern "C" {
    void launch_kernel(int *a, int *b, unsigned int N);
}

// Example class
class CExample{
private:
    int i;

public:
    CExample() : i(0){};

    void run(CudaVector<int> &a, CudaVector<int> &b){
        i += 1;
        launch_kernel(a(), b(), a.Size());
    };

    void run(int* a, int *b, int len){
        i += 1;
        launch_kernel(a, b, len);
    };
};

#endif