#ifndef OPTKIT_OPERATOR_H_
#define OPTKIT_OPERATOR_H_

#include "optkit_abstract_operator.h"
#include "optkit_dense.h"

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

operator * dense_operator_alloc(matrix * A);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_H_ */