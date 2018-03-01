/*******************************************************************************
| Project: CUDA Example                                                        |
| Author:  Bismarck Gomes Souza Junior <bismarck@puc-rio.br>                   |
\******************************************************************************/
#ifndef _CUDAVECTOR_H_
#define _CUDAVECTOR_H_

#include <cuda_runtime.h>
#include <vector>

template <typename T>
class CudaVector {

private:
    T* data;    ///< Variable in device memory
    int size;   ///< Size of device vector

public:
    CudaVector() : size(0) {};

    CudaVector(int size_) { AllocateMemory(size_); };

    CudaVector(int size_, T value) : CudaVector(size_) { SetUniqueValue(value); };

    ~CudaVector() { cudaFree((void *)data); };

    /**
     * Allocate memory in device with a alredy specified class.
     */
    void AllocateMemory() { cudaMalloc((void **)&data, sizeof(T)*size); };

    /**
     * Allocate memory in device with a know size.
     *
     * @param      size_  Size in device
     */
    void AllocateMemory(int size_) { size = size_;  AllocateMemory(); };

    /**
     * Sets the unique value for vector.
     *
     * @param      value  Unique value
     */
    void SetUniqueValue(T value) {
        cudaMemset((void *)data, value, sizeof(T)*size);
    };

    /**
     * Copy from host pointer to device memory.
     *
     * @param      h_data  Host data
     */
    void CopyFromHost(T* h_data) {
        cudaMemcpy(data, h_data, sizeof(T)*size, cudaMemcpyHostToDevice);
    };

    /**
     * Copy from host vector to device memory.
     *
     * @param      h_data  Host data
     */
    void CopyFromHost(std::vector<T> h_data){ CopyFromHost(&h_data[0]); };

    /**
     * Copy from device to host pointer.
     *
     * @param      h_data  Host data
     */
    void CopyToHost(T* h_data){
        cudaMemcpy(&h_data, data, sizeof(T)*size, cudaMemcpyDeviceToHost);
    };

    /**
     * Copy from device to host vector.
     *
     * @param      h_data  Host data
     */
    void CopyToHost(std::vector<T> &h_data) {
        cudaMemcpy(&h_data[0], data, sizeof(T)*size, cudaMemcpyDeviceToHost);
    };

    /**
     * Get vector size.
     *
     * @return     size
     */
    int Size() { return size; };

    /**
     * Get host vector from device memory.
     *
     * @return     host vector
     */
    std::vector<T> Get();

    /**
     * Get device value.
     *
     * @return     device pointer
     */
    T* operator() () { return data; };

    // Overload operators
    CudaVector<T>& operator = (const CudaVector &c1);
    CudaVector<T>& operator = (std::vector<T> c1);
    CudaVector<T>& operator = (T * c1);

    //TODO: implementar ofstream overload operator
};

#endif