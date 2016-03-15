#include "optkit_operator_difference.h"

#ifdef __cplusplus
extern "C" {
#endif

void * difference_operator_data_alloc(size_t offset)
{
	difference_operator_data * op_data;
	op_data = malloc(sizeof(*op_data));
	blas_make_handle(&(op_data->dense_handle));
	op_data->offset = offset;
	op_data->subvec_in.size = 0;
	op_data->subvec_in.stride = 1;
	op_data->subvec_in.data = OK_NULL;
	op_data->subvec_out.size = 0;
	op_data->subvec_out.stride = 1;
	op_data->subvec_out.data = OK_NULL;
	return (void *) op_data;
}

void difference_operator_data_free(void * data)
{
	difference_operator_data * op_data = (differnce_operator_data *) data;
	blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
}

void difference_operator_mul(void * data, vector * input, vector * output)
{
	difference_operator_data * op_data = (difference_operator_data *) data;
	vector_memcpy_vv(input, output);
	op_data->subvec_in.data = input->data + offset;
	op_data->subvec_in.size = input->size - offset;
	op_data->subvec_out.data = output->data;
	op_data->subvec_out.size = output->size - offset;
	blas_axpy(op_data->dense_handle, -kOne, op_data->subvec_in,
		op_data->subvec_out);
}

void difference_operator_mul_t(void * data, vector * input, vector * output)
{
	difference_operator_data * op_data = (difference_operator_data *) data;
	vector_memcpy_vv(input, output);
	op_data->subvec_in.data = input->data;
	op_data->subvec_in.size = input->size - offset;
	op_data->subvec_out.data = output->data + offset;
	op_data->subvec_out.size = output->size - offset;
	blas_axpy(op_data->dense_handle, -kOne, op_data->subvec_in,
		op_data->subvec_out);
}

void difference_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	difference_operator_data * op_data = (difference_operator_data *) data;
	vector_scale(output, beta);
	blas_axpy(op_data->dense_handle, alpha, input, output);
	op_data->subvec_in.data = input->data + offset;
	op_data->subvec_in.size = input->size - offset;
	op_data->subvec_out.data = output->data;
	op_data->subvec_out.size = output->size - offset;
	blas_axpy(op_data->dense_handle, -alpha, op_data->subvec_in,
		op_data->subvec_out);
}

void difference_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	difference_operator_data * op_data;
	op_data = malloc(sizeof(&op_data));
	vector_scale(output, beta);
	blas_axpy(op_data->dense_handle, alpha, input, output);
	op_data->subvec_in.data = input->data;
	op_data->subvec_in.size = input->size - offset;
	op_data->subvec_out.data = output->data + offset;
	op_data->subvec_out.size = output->size - offset;
	blas_axpy(op_data->dense_handle, -alpha, op_data->subvec_in,
		op_data->subvec_out);
}


operator * difference_operator_alloc(size_t n, size_t offset)
{
	return operator_alloc(Difference_Operator,
		A->size1, A->size2,
		difference_operator_data_alloc(n, offset),
		difference_operator_mul,
		difference_operator_mul_t,
		difference_operator_mul_fused,
		difference_operator_mul_t_fused,
		difference_operator_data_free
	);
}

void * block_difference_operator_data_alloc(size_t n_blocks,
	size_t * block_sizes, size_t * offsets)
{
	block_difference_operator_data * op_data;
	op_data = malloc(sizeof(&op_data));
	blas_make_handle(&(op_data->dense_handle));
	op_data->n_blocks = n_blocks;
	op_data->block_sizes = block_sizes;
	op_data->offsets = offsets;
	op_data->subvec_in.size = 0;
	op_data->subvec_in.stride = 1;
	op_data->subvec_in.data = OK_NULL;
	op_data->subvec_out.size = 0;
	op_data->subvec_out.stride = 1;
	op_data->subvec_out.data = OK_NULL;
	return (void *) op_data;
}

void block_difference_operator_data_free(void * data)
{
	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
}

void block_difference_operator_mul(void * data, vector * input, vector * output)
{
	size_t b, block_start = 0;
	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	op_data = malloc(sizeof(&op_data));
	vector_memcpy_vv(input, output);
	for (b = 0; b < op_data->n_blocks; ++b) {
		op_data->subvec_in.data =
			input->data + block_start + op_data->offsets[b];
		op_data->subvec_in.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		op_data->subvec_out.data =
			output->data + block_start;
		op_data->subvec_out.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		blas_axpy(op_data->dense_handle, -kOne, op_data->subvec_in,
			op_data->subvec_out);
		block_start += op_data->block_sizes[b];
	}
}

void block_difference_operator_mul_t(void * data, vector * input,
	vector * output)
{
	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	size_t b, block_start = 0;

	for (b = 0; b < op_data->n_blocks; ++b) {
		op_data->subvec_in.data = input->data + block_start;
		op_data->subvec_in.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		op_data->subvec_out.data =
			output->data + block_start + op_data->offsets[b];
		op_data->subvec_out.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		blas_axpy(op_data->dense_handle, -kOne, op_data->subvec_in,
			op_data->subvec_out);
		block_start += op_data->block_sizes[b];
	}
}

void block_difference_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	size_t b, block_start = 0;

	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	vector_scale(output, beta);
	blas_axpy(op_data->dense_handle, alpha, input, output);

	for (b = 0; b < op_data->n_blocks; ++b) {
		op_data->subvec_in.data =
			input->data + block_start + op_data->offsets[b];
		op_data->subvec_in.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		op_data->subvec_out.data =
			output->data + block_start;
		op_data->subvec_out.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		blas_axpy(op_data->dense_handle, -alpha, op_data->subvec_in,
			op_data->subvec_out);
		block_start += op_data->block_sizes[b];
	}
}

void block_difference_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	vector_scale(output, beta);
	blas_axpy(op_data->dense_handle, alpha, input, output);

	for (b = 0; b < op_data->n_blocks; ++b) {
		op_data->subvec_in.data = input->data + block_start;
		op_data->subvec_in.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		op_data->subvec_out.data =
			output->data + block_start + op_data->offsets[b];
		op_data->subvec_out.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		blas_axpy(op_data->dense_handle, -alpha, op_data->subvec_in,
			op_data->subvec_out);
		block_start += op_data->block_sizes[b];
	}
}


operator * block_difference_operator_alloc(size_t n, size_t n_blocks,
	size_t * block_sizes, size_t * offsets)
{
	return operator_alloc(BlockDifference_Operator,
		n, n,
		block_difference_operator_data_alloc(n_blocks,
			block_sizes, offsets),
		block_difference_operator_mul,
		block_difference_operator_mul_t,
		block_difference_operator_mul_fused,
		block_difference_operator_mul_t_fused,
		block_difference_operator_data_free
	);
}


#ifdef __cplusplus
}
#endif