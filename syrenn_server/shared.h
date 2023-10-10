#ifndef SYRENN_SYRENN_SERVER_SHARED_H_
#define SYRENN_SYRENN_SERVER_SHARED_H_
#include "eigen3/Eigen/Dense"
#include <chrono>
#include <typeinfo>

#define SYRENN_DEVICE_CPU -1

#define PARALLEL_TRANSFORM 1

// #define VERBOSE 1
// #define TIME_TRANSFORM 1
#define TIME_TRANSFORM_TOP 1
// #define TIME_INTERSECTION 1

// #define DEBUG_TRANSFORM 1
// #define debug_plane_idx -1
// #define debug_v1_idx -1
// #define debug_v2_idx -1
// #define debug_face_idx -1

#define TBB_GRAINSIZE_SPLIT 32
#define TBB_GRAINSIZE_CPU_INTERSECT 32
class Timer {
 public:
  Timer() : k(1000) { Reset(); }
  Timer(int k) : k(k) { Reset(); }
  void Reset() { start = std::chrono::system_clock::now(); }
  double Ticks() {
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count() * k;
  }
  std::chrono::time_point<std::chrono::system_clock> start;
  int k;
};

using RMVectorXf = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
// Should only be used for preimage distances
using RMVectorXd = Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>;
using RMVectorXi = Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor>;
using RMMatrixXf = Eigen::Matrix<float, Eigen::Dynamic,
                                 Eigen::Dynamic, Eigen::RowMajor>;
// Should only be used for preimage distances
using RMMatrixXd = Eigen::Matrix<double, Eigen::Dynamic,
                                 Eigen::Dynamic, Eigen::RowMajor>;
using RMMatrixXi = Eigen::Matrix<int, Eigen::Dynamic,
                                 Eigen::Dynamic, Eigen::RowMajor>;


// Helpers to be able to use concurrent_hash_map with IntersectionPointMetadata.
// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x/38140932#38140932
inline void hash_combine(size_t *seed) { }

template <typename T, typename... Rest>
inline void hash_combine(size_t *seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);
  hash_combine(seed, rest...);
}

#endif  // SYRENN_SYRENN_SERVER_SHARED_H_
