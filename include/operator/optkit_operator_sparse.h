#ifndef OPTKIT_OPERATOR_SPARSE_H_
#define OPTKIT_OPERATOR_SPARSE_H_

#include "optkit_abstract_operator.h"
#include "optkit_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sparse_operator_data{
	void * dense_handle;
	void * sparse_handle;
	sp_matrix * A;
} sparse_operator_data;

void * sparse_operator_data_alloc(sp_matrix * A);
void sparse_operator_data_free(void * data);
void sparse_operator_mul(void * data, vector * input, vector * output);
void sparse_operator_mul_t(void * data, vector * input, vector * output);
void sparse_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output);
void sparse_operator_mul_t_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output);

operator * sparse_operator_alloc(sp_matrix * A);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_SPARSE_H_ */