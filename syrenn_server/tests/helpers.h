#include "gtest/gtest.h"
#include "syrenn_server/upolytope.h"
#include "syrenn_server/fullyconnected_transformer.h"

#define PI 3.14159265

std::unique_ptr<FullyConnectedTransformer> RandomFC(size_t in_dims, size_t out_dims);

UPolytope CreateUPolytopeCircleDevice(size_t n_points, float radius);

UPolytope CreateUPolytopeCircleDevice(size_t n_points, size_t n_dims, float radius);

RMMatrixXf CreateUPolytopeCircleMatrix(size_t n_points, float radius);
