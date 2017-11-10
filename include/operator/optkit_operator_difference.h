#ifndef OPTKIT_OPERATOR_DIFFERENCE_H_
#define OPTKIT_OPERATOR_DIFFERENCE_H_

#include "optkit_abstract_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct difference_operator_data{
	void * dense_handle;
	size_t offset;
	vector subvec_in, subvec_out;
} difference_operator_data;

void * difference_operator_data_alloc(size_t offset);
ok_status difference_operator_data_free(void * data);
ok_status difference_operator_mul(void * data, vector * input, vector * output);
ok_status difference_operator_mul_t(void * data, vector * input,
	vector * output);
ok_status difference_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);
ok_status difference_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);

abstract_operator * difference_operator_alloc(sp_matrix * A);

typedef struct block_difference_operator_data{
	void * dense_handle;
	size_t nblocks, * block_sizes, * offsets;
	vector subvec_in, subvec_out;
} difference_operator_data;

ok_status block_difference_operator_data_alloc(size_t n_blocks,
	size_t * block_sizes, size_t * offsets);
ok_status block_difference_operator_data_free(void * data);
ok_status block_difference_operator_mul(void * data, vector * input,
	vector * output);
ok_status block_difference_operator_mul_t(void * data, vector * input,
	vector * output);
ok_status block_difference_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);
ok_status block_difference_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output);

abstract_operator * block_difference_operator_alloc(sp_matrix * A);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_DIFFERENCE_H_ */
