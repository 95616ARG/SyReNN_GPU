#include <iostream>
#include <memory>
#include "syrenn_server/fullyconnected_transformer.h"
#include "cublas_v2.h"

FullyConnectedTransformer::FullyConnectedTransformer(const RMMatrixXf &weights,
                                                     const RMVectorXf &biases,
                                                     const int device)
    : AffineTransformer(device),
      weights_(weights), biases_(biases),
      weights_device_(), biases_device_() {
  if (device_ != -1) {
    CUDAShared::init();
    weights_device_= RMMatrixXfDevice(weights);
    biases_device_= RMMatrixXfDevice(biases);
  }
}

__global__
void SetBiasesKernel(size_t n_rows,
                size_t n_cols,
                float* data,
                const float* biases) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_rows) {
    memcpy(&data[row * n_cols], biases, sizeof(float) * n_cols);
  }
}

void FullyConnectedTransformer::ComputeDevice(RMMatrixXfDevice *inout) const{

#ifdef TIME_FC
  Timer t; t.Reset();
#endif

  int m = inout->rows(),
      k = inout->cols(),
      n = weights_device_.cols();

  CUDAShared::output.resize(m, n);

#ifdef TIME_FC
  std::cerr << t.Ticks() << "ms -- resize\n"; t.Reset();
#endif

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((m / threadsPerBlock.x) + 1);
  // assert (nBlocks.x * threadsPerBlock.x  >= m);
  SetBiasesKernel<<<nBlocks, threadsPerBlock>>>(m, n, CUDAShared::output.data(), biases_device_.data());

#ifdef TIME_FC
  std::cerr << t.Ticks() << "ms -- bias\n"; t.Reset();
#endif

  const float alpha = 1.0f,
              beta  = 1.0f;

  // Because cuBLAS treats matrices as column-major by default, we compute
  // B^T * A^T = (AB)^T in column-major to get AB in row-major.
  // See also: https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
  checkCUBLAS( cublasSgemm(CUDAShared::cublas_handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           n, m, k,
                           &alpha,
                           weights_device_.data(), n,
                           inout->data(), k,
                           &beta,
                           CUDAShared::output.data(), n) );

#ifdef TIME_FC
  std::cerr << t.Ticks() << "ms -- cublasSgemm\n"; t.Reset();
#endif

  inout->swap(CUDAShared::output);
  // CUDAShared::output.resize(0, 0);
  // CUDAShared::output.shrink_to_fit();

#ifdef TIME_FC
  std::cerr << t.Ticks() << "ms -- swap\n"; t.Reset();
#endif

  return;
}
