#include <iostream>
#include <vector>
#include "CudaVector.h"
#include "example.h"



int main(void) {
	// Declare variables
	int len = 10;
	int *b_gpu;
	CudaVector<int> c1(len, 0), c2(len, 0), c3(2, 0);
	std::vector<int> v_host(len, 0), v_host2(len, 0);

	// Create vector in host: v_host
	for (int i = 0; i<len; i++) {
		v_host[i] = i;
	}

	// Copy from host to device
	c2 = v_host;

	// Run in device
	CExample ab = CExample();
	ab.run(c1, c2);

	// Run in device
	//launch_kernel(c1(), c2(), c1.Size());

	// Copy in device
	c3 = c1;

	// Get variable in device memory
	b_gpu = c3();

	// Copy from device to host
	c3.CopyToHost(v_host2);

	// Get from device to host
	v_host2 = c3.Get();
	
	// Print v_host2
	std::cout << "\nCPU:    ";
	for (int i = 0; i<v_host2.size(); i++) {
		std::cout << v_host2[i] << " ";
	}
	std::cout << std::endl;

	return 0;
}