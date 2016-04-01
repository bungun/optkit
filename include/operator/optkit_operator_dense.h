#ifndef OPTKIT_OPERATOR_DENSE_H_
#define OPTKIT_OPERATOR_DENSE_H_

#include "optkit_abstract_operator.h"
#include "optkit_operator_transforms.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dense_operator_data{
	void * dense_handle;
	matrix * A;
} dense_operator_data;

void * dense_operator_data_alloc(matrix * A);
void dense_operator_data_free(void * data);
void dense_operator_mul(void * data, vector * input, vector * output);
void dense_operator_mul_t(void * data, vector * input, vector * output);
void dense_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output);
void dense_operator_mul_t_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output);

operator * dense_operator_alloc(matrix * A);
matrix * dense_operator_get_matrix_pointer(operator * A);

void * dense_operator_export(operator * A);
void * dense_operator_import(operator * A, void * data);
ok_status dense_operator_abs(operator * A);
ok_status dense_operator_pow(operator * A, const ok_float power);
ok_status dense_operator_scale(operator * A, const ok_float scaling);
ok_status dense_operator_scale_left(operator * A, const vector * v);
ok_status dense_operator_scale_right(operator * A, const vector * v);

transformable_operator * dense_operator_to_transformable(operator * A);


#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_DENSE_H_ */
