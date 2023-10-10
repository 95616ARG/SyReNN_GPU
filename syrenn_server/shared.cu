#include "syrenn_server/shared.cuh"

const char* cublasGetErrorString(cublasStatus_t status) {
  switch(status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown error";
}

template <class T>
__host__
RMMatrixXTDevice<T>::RMMatrixXTDevice()
  : n_rows_(0), n_cols_(0) {}

template <class T>
__host__
RMMatrixXTDevice<T>::RMMatrixXTDevice(size_t capacity)
  : n_rows_(0), n_cols_(0) {
    data_.resize(capacity);
  }

template <class T>
__host__
RMMatrixXTDevice<T>::RMMatrixXTDevice(size_t n_rows, size_t n_cols)
  : data_(n_rows * n_cols), n_rows_(n_rows), n_cols_(n_cols) {}

template <class T>
__host__
RMMatrixXTDevice<T>::RMMatrixXTDevice(size_t n_rows, size_t n_cols, size_t capacity)
  : data_(capacity), n_rows_(n_rows), n_cols_(n_cols) {}

template <class T>
__host__
RMMatrixXTDevice<T>::RMMatrixXTDevice(const RMMatrixXTDevice<T>::RMMatrixXT &rhs)
  : data_(rhs.data(), rhs.data() + rhs.size()),
    n_rows_(rhs.rows()), n_cols_(rhs.cols()) {}

template <class T>
__host__
RMMatrixXTDevice<T>::RMMatrixXTDevice(const RMMatrixXTDevice<T>::RMMatrixXT &rhs, size_t capacity)
  : data_(rhs.data(), rhs.data() + rhs.size()),
    n_rows_(rhs.rows()), n_cols_(rhs.cols()) {
    data_.resize(capacity);
  }

template <class T>
__host__
RMMatrixXTDevice<T>::RMMatrixXTDevice(const RMMatrixXTDevice<T> &rhs)
  : data_(rhs.device().begin(), rhs.device().begin() + rhs.size()),
    n_rows_(rhs.rows()), n_cols_(rhs.cols()) {}

template <class T>
__host__
RMMatrixXTDevice<T>::RMMatrixXTDevice(const RMMatrixXTDevice<T> &rhs, size_t capacity)
  : data_(capacity),
    n_rows_(rhs.rows()), n_cols_(rhs.cols()) {
  thrust::copy(rhs.device().begin(),
               rhs.device().begin() + rhs.size(),
               device().begin());
}

// template <class T>
// __host__ RMMatrixXTDevice<T>::~RMMatrixXTDevice() {
//   data_.thrust::~device_vector<T>();
// }

template <class T>
__host__
size_t RMMatrixXTDevice<T>::rows() const { return n_rows_; };

template <class T>
__host__
size_t RMMatrixXTDevice<T>::cols() const { return n_cols_; };

template <class T>
__host__
size_t RMMatrixXTDevice<T>::size() const { return n_rows_ * n_cols_; }

template <class T>
__host__
size_t RMMatrixXTDevice<T>::capacity() const { return data_.capacity(); }

template <class T>
__host__
void RMMatrixXTDevice<T>::fill(const T x) {
  thrust::fill_n(data_.begin(), size(), x);
}

template <class T>
__host__
void RMMatrixXTDevice<T>::fill(const RMMatrixXTDevice<T>::RMMatrixXT &rhs) {
  resize(rhs.rows(), rhs.cols());
  thrust::copy(rhs.data(),
               rhs.data() + rhs.size(),
               device().begin());
}

template <class T>
__host__
void RMMatrixXTDevice<T>::fill_n(const T x, size_t n) {
  thrust::fill_n(data_.begin(), n, x);
}

template <class T>
__host__
void RMMatrixXTDevice<T>::fill_n_rows(const T x, size_t n_rows) {
  thrust::fill_n(data_.begin(), n_rows * cols(), x);
}

template <class T>
__host__
void RMMatrixXTDevice<T>::reserve_rows(size_t n) {
  set_capacity((n_rows_ + n) * n_cols_);
}

template <class T>
__host__
void RMMatrixXTDevice<T>::clear() {
  data_.clear();
  n_rows_ = 0;
  n_cols_ = 0;
}


template <class T>
__host__
void RMMatrixXTDevice<T>::set_capacity(size_t capacity) {
  if (data_.size() < capacity) {
    // std::cerr << "capacity " << data_.size() << " --> " << capacity << " ... ";
    data_.resize(capacity);
    // std::cerr << "done!\n";
  }
}

template <class T>
__host__
void RMMatrixXTDevice<T>::resize_rows(size_t n) {
  set_capacity(n * cols());
  n_rows_ = n;
}

template <class T>
__host__
void RMMatrixXTDevice<T>::resize(size_t n_rows, size_t n_cols) {
  n_cols_ = n_cols;
  resize_rows(n_rows);
}

template <class T>
__host__
void RMMatrixXTDevice<T>::shrink_to_fit() {
  data_.resize(size());
  data_.shrink_to_fit();
}

template <class T>
__host__
T* RMMatrixXTDevice<T>::data() {
  return thrust::raw_pointer_cast(data_.data());
};

template <class T>
__host__
const T* RMMatrixXTDevice<T>::data() const {
  return thrust::raw_pointer_cast(data_.data());
};

template <class T>
__host__
void RMMatrixXTDevice<T>::swap(thrust::device_vector<T>& rhs, size_t n_rows, size_t n_cols) {
  // assert (rhs.size() == n_rows * n_cols);
  n_rows_ = n_rows;
  n_cols_ = n_cols;
  // Retain the original capacity.
  if (rhs.size() < data_.capacity()) { rhs.resize(data_.capacity()); }
  data_.swap(rhs);
  return;
}

template <class T>
__host__
void RMMatrixXTDevice<T>::swap(RMMatrixXTDevice<T>& rhs) {
  std::swap(n_rows_, rhs.n_rows_);
  std::swap(n_cols_, rhs.n_cols_);
  // rhs.data_.resize(data_.capacity());
  data_.swap(rhs.data_);
  return;
}

template <class T>
__host__
thrust::device_vector<T> & RMMatrixXTDevice<T>::device() {
  return data_;
}

template <class T>
__host__
const thrust::device_vector<T> & RMMatrixXTDevice<T>::device() const {
  return data_;
}

template <class T>
__host__
thrust::host_vector<T> RMMatrixXTDevice<T>::host() const {
  return data_;
}

template <class T>
__host__
RMMatrixXTDevice<T>::RMMatrixXT RMMatrixXTDevice<T>::eigen() const {
  return Eigen::Map<const RMMatrixXTDevice<T>::RMMatrixXT>(
    thrust::raw_pointer_cast(host().data()),
    rows(), cols());
}

template <class T>
__host__
RMMatrixXTDevice<T>::RMVectorXT RMMatrixXTDevice<T>::row(size_t idx) const {
  std::vector<T> row_host(cols());
  thrust::copy(
    device().begin() + idx * cols(),
    device().begin() + (idx + 1) * cols(),
    row_host.begin()
  );
  return Eigen::Map<const RMMatrixXTDevice<T>::RMVectorXT>(
    row_host.data(),
    cols());
}

template <class T>
__host__
T RMMatrixXTDevice<T>::at(size_t row_idx, size_t col_idx) const {
  return data_[row_idx * cols() + col_idx];
}

template <class T>
__host__
std::ostream& operator<<(std::ostream& os, const RMMatrixXTDevice<T>& rhs) {
  os << rhs.eigen();
  return os;
}

template<>
__host__ void RMMatrixXTDevice<float>::matmul(const RMMatrixXTDevice<float>& rhs) {

  int m = rows(),
      k = cols(),
      n = rhs.cols();

  CUDAShared::output.resize(m, n);

  const float alpha = 1.0f,
              beta  = 0.0f;

  // Because cuBLAS treats matrices as column-major by default, we compute
  // B^T * A^T = (AB)^T in column-major to get AB in row-major.
  // See also: https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
  checkCUBLAS( cublasSgemm(CUDAShared::cublas_handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           n, m, k,
                           &alpha,
                           rhs.data(), n,
                           data(), k,
                           &beta,
                           CUDAShared::output.data(), n) );

  this->swap(CUDAShared::output);

  return;
}

template <>
__host__ RMMatrixXTDevice<float>& RMMatrixXTDevice<float>::operator*=(const RMMatrixXTDevice<float>& rhs) {
  this->matmul(rhs);
  return *this;
}

template <class T>
__host__ T RMMatrixXTDevice<T>::mean() {
  T sum = thrust::reduce(device().begin(), device().end(), (T)0, thrust::plus<T>());
  return sum/(T)(device().size());
}

template <class T>
__global__ void MeanByAxix0Kernel(const size_t n_rows,
                                  const size_t n_cols,
                                  const T* data,
                                  T* result) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n_cols) {
    T sum = 0.;
    for (int row = 0; row < n_rows; row++) {
      sum += data[row * n_cols + col];
    }
    result[col] = sum / (T)n_rows;
  }
}

template <class T>
__global__ void MeanByAxix1Kernel(const size_t n_rows,
                                  const size_t n_cols,
                                  const T* data,
                                  T* result) {
  const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_rows) {
    T sum = 0.;
    for (int col = 0; col < n_cols; col++) {
      sum += data[row * n_cols + col];
    }
    result[row] = sum / (T)n_cols;
  }
}

template <class T>
__host__ RMMatrixXTDevice<T> RMMatrixXTDevice<T>::mean(int axis) {
  if (axis == 0) {
    RMMatrixXTDevice<T> result(cols(), 1);

    const dim3 threadsPerBlock(1024);
    const dim3 nBlocks((rows() / threadsPerBlock.x) + 1);
    MeanByAxix0Kernel<T><<<nBlocks, threadsPerBlock>>>(
      rows(),
      cols(),
      data(),
      result.data()
    );

    return result;

  }

  if (axis == 1) {
    RMMatrixXTDevice<T> result(rows(), 1);

    const dim3 threadsPerBlock(1024);
    const dim3 nBlocks((cols() / threadsPerBlock.x) + 1);
    MeanByAxix1Kernel<T><<<nBlocks, threadsPerBlock>>>(
      rows(),
      cols(),
      data(),
      result.data()
    );

    return result;

  }

  assert (false);
}

template <class T>
__global__ void ArgMaxKernel(const size_t n_rows,
                             const size_t n_cols,
                             const T* data,
                             int* labels) {
  const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_rows) {
    T max = data[row * n_cols];
    labels[row] = 0;
    for (int idx = 1; idx < n_cols; idx++) {
      if (data[row * n_cols + idx] > max) {
        max = data[row * n_cols + idx];
        labels[row] = idx;
      }
    }
  }
}

template <class T>
__host__ RMMatrixXTDevice<int> RMMatrixXTDevice<T>::argmax() {

  RMMatrixXTDevice<int> labels(rows(), 1);

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((rows() / threadsPerBlock.x) + 1);
  ArgMaxKernel<T><<<nBlocks, threadsPerBlock>>>(
    rows(),
    cols(),
    data(),
    labels.data()
  );

  return labels;
}

cublasHandle_t CUDAShared::cublas_handle;
cudnnHandle_t CUDAShared::cudnn_handle;
RMMatrixXfDevice CUDAShared::output;
RMMatrixXfDevice CUDAShared::workspace;
bool CUDAShared::is_inited = false;

void CUDAShared::init(const int device) {
  if (is_inited == false) {

    checkCudaErrors( cudaSetDevice(device) );
    checkCUBLAS( cublasCreate(&cublas_handle) );
    checkCUDNN( cudnnCreate(&cudnn_handle) );

    output.resize(1, 1);
    output *= output;
    output.resize(1200000, 200);

    is_inited = true;
  }
}

void CUDAShared::finish() {
  if (is_inited) {

    checkCUBLAS( cublasDestroy(cublas_handle) );
    checkCUDNN( cudnnDestroy(cudnn_handle) );

    // output.device().clear();
    // output.device().shrink_to_fit();
    // output.~RMMatrixXfDevice();

    // workspace.device().clear();
    // workspace.device().shrink_to_fit();
    // workspace.~RMMatrixXfDevice();

    is_inited = false;
  }
}
