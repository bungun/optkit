#ifndef OPTKIT_OPERATOR_H_GUARD
#define OPTKIT_OPERATOR_H_GUARD

#include "optkit_defs.h"
#include "optkit_dense.h"
#include "optkit_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum OPERATOR{
	Id_Operator = 101,
	Neg_Operator = 102,
	Add_Operator = 103,
	Cat_Operator = 104,
	Split_Operator = 105, 
	Dense_Operator = 201,
	SparseCSR_Operator = 301,
	SparseCSC_Operator = 302,
	SparseCOO_Operator = 303,
	Diagonal_Operator = 401,
	Banded_Operator = 402,
	Triangular_Operator = 403,
	Kronecker_Operator = 404,
	Toeplitz_Operator = 405,
	Circulant_Operator = 406,
	Convolution_Operator = 501,
	CircularConvolution_Operator = 502,
	Fourier_Operator = 503,
	Difference_Operator = 504,
	Upsampling_Operator = 505,
	Downsampling_Operator = 506,
	DirectProjection_Operator = 601,
	IndirectProjection_Operator = 602,
	Other_Operator = 1000
} OPTKIT_OPERATOR;

typedef struct abstract_linear_operator{
	size_t size1, size2;
	void * data;
	void (* apply)(void * data, vector * input, vector * output);
	void (* adjoint)(void * data, vector * input, vector * output);
	void (* fused_apply)(void * data, ok_float alpha, vector * input,
		ok_float beta, vector * output);
	void (* fused_adjoint)(void * data, ok_float alpha, vector * input,
		ok_float beta, vector * output);
	OPTKIT_OPERATOR kind;
} operator_t;

typedef struct dense_operator_data{
	void * dense_handle;
	matrix * A;
} dense_operator_data;

void dense_operator_data_alloc(void ** data);
void dense_operator_data_free(void * data);
void dense_operator_mul(void * data, vector * input, vector * output);
void dense_operator_mul_t(void * data, vector * input, vector * output);
void dense_operator_mul_fused(void * data, ok_float alpha, 
	vector * input, ok_float beta, vector * output);
void dense_operator_mul_t_fused(void * data, ok_float alpha, 
	vector * input, ok_float beta, vector * output);

operator_t * dense_operator_alloc(matrix * A);

typedef struct sparse_operator_data{
	void * dense_handle;
	void * sparse_handle;
	sp_matrix * A;
} sparse_operator_data;

void sparse_operator_data_alloc(void ** data);
void sparse_operator_data_free(void * data);
void sparse_operator_mul(void * data, vector * input, vector * output);
void sparse_operator_mul_t(void * data, vector * input, vector * output);
void sparse_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);
void sparse_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);

operator_t * sparse_operator_alloc(sp_matrix * A);

void operator_free(operator_t * op);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_H_GUARD */