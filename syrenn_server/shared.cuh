#ifndef SYRENN_SYRENN_SERVER_SHARED_CUDA_H_
#define SYRENN_SYRENN_SERVER_SHARED_CUDA_H_
#include "syrenn_server/shared.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "cuDNN failure\nError: " << cudnnGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure\nError: " << cudaGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}

const char* cublasGetErrorString(cublasStatus_t status);

#define checkCUBLAS(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                              \
      _error << "cuBLAS failure\nError: " << cublasGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}

// CUDA Utility Helper Functions

static void showDevices( void )
{
    int totalDevices;
    checkCudaErrors(cudaGetDeviceCount( &totalDevices ));
    printf("\nThere are %d CUDA capable devices on your machine :\n", totalDevices);
    for (int i=0; i< totalDevices; i++) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties( &prop, i ));
        printf( "device %d : sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, boardGroupID=%d\n",
                    i, prop.multiProcessorCount, prop.major, prop.minor,
                    (float)prop.clockRate*1e-3,
                    (int)(prop.totalGlobalMem/(1024*1024)),
                    (float)prop.memoryClockRate*1e-3,
                    prop.ECCEnabled,
                    prop.multiGpuBoardGroupID);
    }
}

template <class T> class RMMatrixXTDevice;
template <class T>
__host__ std::ostream& operator<<(std::ostream& os, const RMMatrixXTDevice<T>& data);

// Thrust-based RMMartixXf on device, which emulates Eigen-based RMMatrixXf.
template <class T>
class RMMatrixXTDevice {
public:
  typedef Eigen::Matrix<T, 1             , Eigen::Dynamic, Eigen::RowMajor> RMVectorXT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMMatrixXT;

  __host__ RMMatrixXTDevice();
  __host__ RMMatrixXTDevice(size_t capacity);

  __host__ RMMatrixXTDevice(size_t n_rows, size_t n_cols);
  __host__ RMMatrixXTDevice(size_t n_rows, size_t n_cols, size_t capacity);

  __host__ RMMatrixXTDevice(const RMMatrixXT &rhs);
  __host__ RMMatrixXTDevice(const RMMatrixXT &rhs, size_t capacity);

  __host__ RMMatrixXTDevice(const RMMatrixXTDevice<T> &rhs);
  __host__ RMMatrixXTDevice(const RMMatrixXTDevice<T> &rhs, size_t capacity);

  // __host__ ~RMMatrixXTDevice();

  // Returns the number of rows, cols, size (i.e., rows * cols) and capacity.
  __host__ size_t rows() const;
  __host__ size_t cols() const;
  __host__ size_t size() const;
  __host__ size_t capacity() const;

  // Fills the matrix with given size and value.
  __host__ void fill(const T x);
  __host__ void fill(const RMMatrixXT & rhs);
  __host__ void fill_n(const T x, size_t n);
  __host__ void fill_n_rows(const T x, size_t n_rows);

  // Clears the matrix as well as sets rows, cols, and size to zero.
  // NOTE: The capacity and content after clear is _unspecified_.
  __host__ void clear();

  // Resizes the matrix.
  __host__ void resize_rows(size_t n);
  __host__ void resize(size_t n_rows, size_t n_cols);
  __host__ void reserve_rows(size_t n);
  __host__ void set_capacity(size_t n);
  __host__ void shrink_to_fit();

  // Swaps the underlying data with given RMMatrixXfDevice or vector.
  __host__ void swap(thrust::device_vector<T>& rhs, size_t n_rows, size_t n_cols);
  __host__ void swap(RMMatrixXTDevice<T>& rhs);

  // Returns the (const) raw pointer to the device memory array used internally.
  __host__ T* data();
  __host__ const T* data() const;

  // Returns the (const) reference to the internal Thrust vector on device.
  __host__ thrust::device_vector<T> &device();
  __host__ const thrust::device_vector<T> & device() const;

  // Returns a copy of the internal Thrust device vector on host.
  __host__ thrust::host_vector<T> host() const;

  // Returns a copy of the internal Thrust device vector on host as RMMartixXf.
  __host__ RMMatrixXT eigen() const;

  // Returns a copy of the given row on host as RMMartixXf..
  __host__ RMVectorXT row(size_t idx) const;

  // Returns a copy of the data at the given position on host.
  __host__ T at(size_t row_idx, size_t col_idx) const;

  // Prints the underlying matrix, callable from host.
  __host__ friend std::ostream& operator<<(std::ostream& os, const RMMatrixXTDevice<T>& data);

  __host__ void matmul(const RMMatrixXTDevice<float>& rhs);

  __host__ RMMatrixXTDevice<float>& operator*=(const RMMatrixXTDevice<float>& rhs);

  __host__ T mean();
  __host__ RMMatrixXTDevice<T> mean(int axis);
  __host__ RMMatrixXTDevice<int> argmax();

private:
  thrust::device_vector<T> data_;
  size_t n_rows_;
  size_t n_cols_;
};

template class RMMatrixXTDevice<float>;
template class RMMatrixXTDevice<double>;
template class RMMatrixXTDevice<int>;
using RMMatrixXfDevice = RMMatrixXTDevice<float>;
using RMMatrixXdDevice = RMMatrixXTDevice<double>;
using RMMatrixXiDevice = RMMatrixXTDevice<int>;

class CUDAShared {
public:
  static void init(const int device = 0);
  static void finish();

  static cublasHandle_t cublas_handle;
  static cudnnHandle_t cudnn_handle;

  static RMMatrixXfDevice output;
  static RMMatrixXfDevice workspace;

  static bool is_inited;
};

#endif  // SYRENN_SYRENN_SERVER_SHARED_CUDA_H_
