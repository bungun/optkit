#ifndef OPTKIT_OPERATOR_SPARSE_H_
#define OPTKIT_OPERATOR_SPARSE_H_

#include "optkit_abstract_operator.h"
#include "optkit_operator_transforms.h"
#include "optkit_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sparse_operator_data{
	void * sparse_handle;
	sp_matrix * A;
} sparse_operator_data;

void * sparse_operator_data_alloc(sp_matrix * A);
ok_status sparse_operator_data_free(void * data);
ok_status sparse_operator_mul(void * data, vector * input, vector * output);
ok_status sparse_operator_mul_t(void * data, vector * input, vector * output);
ok_status sparse_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);
ok_status sparse_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);

abstract_operator * sparse_operator_alloc(sp_matrix * A);
sp_matrix * sparse_operator_get_matrix_pointer(abstract_operator * A);

void * sparse_operator_export(abstract_operator * A);
void * sparse_operator_import(abstract_operator * A, void * data);
ok_status sparse_operator_abs(abstract_operator * A);
ok_status sparse_operator_pow(abstract_operator * A, const ok_float power);
ok_status sparse_operator_scale(abstract_operator * A, const ok_float scaling);
ok_status sparse_operator_scale_left(abstract_operator * A, const vector * v);
ok_status sparse_operator_scale_right(abstract_operator * A, const vector * v);

transformable_operator * sparse_operator_to_transformable(abstract_operator * A);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_SPARSE_H_ */
