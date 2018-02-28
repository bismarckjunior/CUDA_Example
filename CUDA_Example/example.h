#ifndef _CEXAMPLE_H_
#define _CEXAMPLE_H_

#include "CudaVector.h"

// Launch kernels
extern "C" {	
	void launch_kernel(int *a, int *b, unsigned int N);
}


class CExample{
private:
	int i;

public:
	CExample();

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