#ifndef OPTKIT_OPERATOR_DIAGONAL_H_
#define OPTKIT_OPERATOR_DIAGONAL_H_

#include "optkit_abstract_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diagonal_operator_data{
	void * dense_handle;
	vector * d;
} diagonal_operator_data;

void * diagonal_operator_data_alloc(vector * d);
ok_status diagonal_operator_data_free(void * data);
ok_status diagonal_operator_mul(void * data, vector * input, vector * output);
ok_status diagonal_operator_mul_t(void * data, vector * input, vector * output);
ok_status diagonal_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);
ok_status diagonal_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);

operator * diagonal_operator_alloc(vector * d);

ok_status diagonal_operator_abs(operator * A);
ok_status diagonal_operator_pow(operator * A, const ok_float power);
ok_status diagonal_operator_scale(operator * A, const ok_float scaling);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_DIAGONAL_H_ */
