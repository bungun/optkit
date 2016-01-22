#ifndef OPTKIT_OPERATOR_H_GUARD
#define OPTKIT_OPERATOR_H_GUARD

#include "optkit_abstract_operator.h"
#include "optkit_dense.h"
#include "optkit_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

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