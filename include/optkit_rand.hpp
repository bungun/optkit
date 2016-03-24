#ifndef OPTKIT_RAND_H_
#define OPTKIT_RAND_H_

#include "optkit_defs.h"

#ifdef __CUDACC__
#include <curand_kernel.h>
#include "optkit_defs_gpu.h"

namespace {
	__global__ void setup_kernel(curandState * state, unsigned long seed)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		curand_init(seed, tid, 0, &state[tid]);
	}

	__global__ void generate(curandState * globalState, ok_float * data,
		size_t size)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x, i;
		for (i = tid; i < size; i += gridDim.x * blockDim.x)
			data[i] = curand_uniform_double(&globalState[tid]);
	}
}

void __ok_rand(ok_float * x, size_t size) {
	size_t num_rand = size <= kMaxGridSize ? size : kMaxGridSize;
	curandState * devStates;
	int grid_dim;

	ok_alloc_gpu(devStates, num_rand * sizeof(*devStates));
	grid_dim = calc_grid_dim(num_rand, kBlockSize);

	__setup_kernel<<<grid_dim, kBlockSize>>>(devStates, 0);
	__generate<<<grid_dim, kBlockSize>>>(devStates, x, size);

	cudaFree(devStates);
}
#else
#include <random>

void __ok_rand(ok_float * x, size_t size)
{
	size_t i;
	std::default_random_engine generator;
	std::uniform_real_distribution<ok_float> dist(kZero, kOne);

	for (i = 0; i < size; ++i)
		x[i] = dist(generator);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void ok_rand(ok_float * x, size_t size)
{
	__rand(x, size);
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_RAND_H_ */