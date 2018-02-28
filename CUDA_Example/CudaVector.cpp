#include "CudaVector.h"

template <typename T>
std::vector<T> CudaVector<T>::Get(){
	std::vector<T> host(size);
	CopyToHost(host);
	return host;
}

template <typename T>
CudaVector<T>& CudaVector<T>::operator = (std::vector<T> c1){
	CopyFromHost(c1);
	return *this;
}

template <typename T>
CudaVector<T>& CudaVector<T>::operator = (T* c1){
	CopyFromHost(c1);
	return *this;
}

template <typename T>
CudaVector<T>& CudaVector<T>::operator = (const CudaVector &c1){
	if (size != c1.size)
		AllocateMemory(c1.size);

	cudaMemcpy(data, c1.data, size*sizeof(T), cudaMemcpyDeviceToDevice);

	return *this;
}

template class CudaVector<int>;
template class CudaVector<float>;
