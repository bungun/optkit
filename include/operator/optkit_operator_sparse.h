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
sp_matrix * sparse_operator_get_matrix_pointer(operator * A);

void * sparse_operator_export(operator * A);
ok_status sparse_operator_import(operator * A, void * data);
ok_status sparse_operator_abs(operator * A);
ok_status sparse_operator_pow(operator * A, const ok_float power);
ok_status sparse_operator_scale(operator * A, const ok_float scaling);
ok_status sparse_operator_scale_left(operator * A, const vector * v);
ok_status sparse_operator_scale_right(operator * A, const vector * v);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_SPARSE_H_ */