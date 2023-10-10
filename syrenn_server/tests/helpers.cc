#include "gtest/gtest.h"
#include "syrenn_server/upolytope.h"
#include "syrenn_server/tests/helpers.h"
#include "syrenn_server/fullyconnected_transformer.h"

std::unique_ptr<FullyConnectedTransformer> RandomFC(size_t in_dims, size_t out_dims) {
  RMMatrixXf weights(in_dims, out_dims);
  weights.setRandom();
  RMMatrixXfDevice weights_device(weights);

  RMVectorXf biases(1, out_dims);
  biases.setRandom();
  RMMatrixXfDevice biases_device(biases);

  return std::make_unique<FullyConnectedTransformer>(weights_device, biases_device);
}

UPolytope CreateUPolytopeCircleDevice(size_t n_points, float radius) {
  float rad = 2. * PI / (n_points);
  float rad_shift = 2. * PI / 20.;

  std::vector<size_t> polytope_vector(n_points, 0);
  RMMatrixXf vertices(n_points, 2);
  for (int i = 0; i < n_points; i++) {
    vertices.row(i) << radius * sin(i * rad + rad_shift), radius * cos(i * rad + rad_shift);
    polytope_vector[i] = i;
  }
  RMMatrixXf vertices_device(vertices);
  return UPolytope(&vertices_device, 2, {polytope_vector});
}

UPolytope CreateUPolytopeCircleDevice(size_t n_points, size_t n_dims, float radius) {
  UPolytope inout = CreateUPolytopeCircleDevice(n_points, radius);
  RandomFC(2, n_dims)->TransformUPolytope(&inout);
  return inout;
}

RMMatrixXf CreateUPolytopeCircleMatrix(size_t n_points, float radius) {
  float rad = 2. * PI / (n_points);
  float rad_shift = 2. * PI / 20.;

  RMMatrixXf vertices(n_points, 2);
  for (int i = 0; i < n_points; i++) {
    vertices.row(i) << radius * sin(i * rad + rad_shift), radius * cos(i * rad + rad_shift);
  }
  return vertices;

}
