#include "optkit_operator_difference.h"

#ifdef __cplusplus
extern "C" {
#endif

void * difference_operator_data_alloc(size_t offset)
{
	ok_status err = OPTKIT_SUCCESS;
	difference_operator_data * op_data;
	ok_alloc(op_data, sizeof(*op_data));
	err = OK_SCAN_ERR( blas_make_handle(&(op_data->dense_handle)) );
	op_data->offset = offset;
	op_data->subvec_in.size = 0;
	op_data->subvec_in.stride = 1;
	op_data->subvec_in.data = OK_NULL;
	op_data->subvec_out.size = 0;
	op_data->subvec_out.stride = 1;
	op_data->subvec_out.data = OK_NULL;
	if (err) {
		difference_operator_data_free(op_data);
		op_data = OK_NULL;
	}

	return (void *) op_data;
}

ok_status difference_operator_data_free(void * data)
{
	difference_operator_data * op_data = (differnce_operator_data *) data;
	OK_CHECK_PTR(op_data);
	ok_status err = blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
	return OK_SCAN_ERR( err );
}

ok_status difference_operator_mul(void * data, vector * input, vector * output)
{
	difference_operator_data * op_data = (difference_operator_data *) data;
	OK_RETURNIF_ERR( vector_memcpy_vv(input, output) );
	op_data->subvec_in.data = input->data + offset;
	op_data->subvec_in.size = input->size - offset;
	op_data->subvec_out.data = output->data;
	op_data->subvec_out.size = output->size - offset;
	return blas_axpy(op_data->dense_handle, -kOne, op_data->subvec_in,
		op_data->subvec_out);
}

ok_status difference_operator_mul_t(void * data, vector * input, vector * output)
{
	difference_operator_data * op_data = (difference_operator_data *) data;
	OK_RETURNIF_ERR( vector_memcpy_vv(input, output) );
	op_data->subvec_in.data = input->data;
	op_data->subvec_in.size = input->size - offset;
	op_data->subvec_out.data = output->data + offset;
	op_data->subvec_out.size = output->size - offset;
	return blas_axpy(op_data->dense_handle, -kOne, op_data->subvec_in,
		op_data->subvec_out);
}

ok_status difference_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	difference_operator_data * op_data = (difference_operator_data *) data;
	OK_RETURNIF_ERR( vector_scale(output, beta) );
	OK_RETURNIF_ERR( blas_axpy(op_data->dense_handle, alpha, input, output) );
	op_data->subvec_in.data = input->data + offset;
	op_data->subvec_in.size = input->size - offset;
	op_data->subvec_out.data = output->data;
	op_data->subvec_out.size = output->size - offset;
	return blas_axpy(op_data->dense_handle, -alpha, op_data->subvec_in,
		op_data->subvec_out);
}

ok_status difference_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	difference_operator_data * op_data = (difference_operator_data *) data;
	OK_RETURNIF_ERR( vector_scale(output, beta) );
	OK_RETURNIF_ERR( blas_axpy(op_data->dense_handle, alpha, input, output) );
	op_data->subvec_in.data = input->data;
	op_data->subvec_in.size = input->size - offset;
	op_data->subvec_out.data = output->data + offset;
	op_data->subvec_out.size = output->size - offset;
	return blas_axpy(op_data->dense_handle, -alpha, op_data->subvec_in,
		op_data->subvec_out);
}


abstract_operator * difference_operator_alloc(size_t n, size_t offset)
{
	abstract_operator * o = OK_NULL;
	void * data = difference_operator_data_alloc(n, offset);

	if (data) {
		ok_alloc(o, sizeof(*o));
		o->kind = OkOperatorDifference;
		o->size1 = n;
		o->size2 = n;
		o->data = data;
		o->apply = difference_operator_mul;
		o->adjoint = difference_operator_mul_t;
		o->fused_apply = difference_operator_mul_fused;
		o->fused_adjoint = difference_operator_mul_t_fused;
		o->free = difference_operator_data_free;
	}
	return o;
}

void * block_difference_operator_data_alloc(size_t n_blocks,
	size_t * block_sizes, size_t * offsets)
{
	ok_status err = OPTKIT_SUCCESS;
	block_difference_operator_data * op_data = OK_NULL;

	if (block_sizes && offsets) {
		ok_alloc(op_data, sizeof(&op_data));
		err = OK_SCAN_ERR( blas_make_handle(&(op_data->dense_handle)) );
		op_data->n_blocks = n_blocks;
		op_data->block_sizes = block_sizes;
		op_data->offsets = offsets;
		op_data->subvec_in.size = 0;
		op_data->subvec_in.stride = 1;
		op_data->subvec_in.data = OK_NULL;
		op_data->subvec_out.size = 0;
		op_data->subvec_out.stride = 1;
		op_data->subvec_out.data = OK_NULL;
		if (err) {
			block_difference_operator_data_free(op_data);
			op_data = OK_NULL;
		}
	}
	return (void *) op_data;
}

ok_status block_difference_operator_data_free(void * data)
{
	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	OK_CHECK_PTR(op_data);
	ok_status err = blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
	return OK_SCAN_ERR( err );
}

ok_status block_difference_operator_mul(void * data, vector * input,
	vector * output)
{
	size_t b, block_start = 0;
	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	OK_CHECK_PTR(op_data);

	for (b = 0; b < op_data->n_blocks; ++b) {
		op_data->subvec_in.data =
			input->data + block_start + op_data->offsets[b];
		op_data->subvec_in.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		op_data->subvec_out.data =
			output->data + block_start;
		op_data->subvec_out.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		OK_CHECK_ERR( err, blas_axpy(op_data->dense_handle, -kOne,
			op_data->subvec_in, op_data->subvec_out) );
		block_start += op_data->block_sizes[b];
	}
	return err;
}

ok_status block_difference_operator_mul_t(void * data, vector * input,
	vector * output)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t b, block_start = 0;
	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	OK_CHECK_PTR(op_data);

	for (b = 0; b < op_data->n_blocks && !err; ++b) {
		op_data->subvec_in.data = input->data + block_start;
		op_data->subvec_in.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		op_data->subvec_out.data =
			output->data + block_start + op_data->offsets[b];
		op_data->subvec_out.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		OK_CHECK_ERR( err, blas_axpy(op_data->dense_handle, -kOne,
			op_data->subvec_in, op_data->subvec_out) );
		block_start += op_data->block_sizes[b];
	}
	return err;
}

ok_status block_difference_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t b, block_start = 0;
	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	OK_CHECK_PTR(op_data);
	OK_RETURNIF_ERR( vector_scale(output, beta) );
	OK_RETURNIF_ERR( blas_axpy(op_data->dense_handle, alpha, input,
		output) );

	for (b = 0; b < op_data->n_blocks && !err; ++b) {
		op_data->subvec_in.data =
			input->data + block_start + op_data->offsets[b];
		op_data->subvec_in.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		op_data->subvec_out.data =
			output->data + block_start;
		op_data->subvec_out.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		OK_CHECK_ERR( err, blas_axpy(op_data->dense_handle, -alpha,
			op_data->subvec_in, op_data->subvec_out) );
		block_start += op_data->block_sizes[b];
	}
	return err;
}

ok_status block_difference_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t b, block_start = 0;
	block_difference_operator_data * op_data =
		(block_difference_operator_data *) data;
	OK_CHECK_PTR(op_data);
	OK_RETURNIF_ERR( vector_scale(output, beta) );
	OK_RETURNIF_ERR( blas_axpy(op_data->dense_handle, alpha, input,
		output) );

	for (b = 0; b < op_data->n_blocks && !err; ++b) {
		op_data->subvec_in.data = input->data + block_start;
		op_data->subvec_in.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		op_data->subvec_out.data =
			output->data + block_start + op_data->offsets[b];
		op_data->subvec_out.size =
			op_data->block_sizes[b] - op_data->offsets[b];
		OK_CHECK_ERR( err, blas_axpy(op_data->dense_handle, -alpha,
			op_data->subvec_in, op_data->subvec_out) );
		block_start += op_data->block_sizes[b];
	}
	return err;
}


abstract_operator * block_difference_operator_alloc(size_t n, size_t n_blocks,
	size_t * block_sizes, size_t * offsets)
{
	abstract_operator * o = OK_NULL;
	void * data = OK_NULL;
	if (block_sizes && offsets) {
		data = block_difference_operator_data_alloc(n_blocks,
			block_sizes, offsets);
		if (data) {
			ok_alloc(o, sizeof(*o));
			o->kind = OkOperatorBlockDifference;
			o->size1 = n;
			o->size2 = n;
			o->data = data;
			o->apply = block_difference_operator_mul;
			o->adjoint = block_difference_operator_mul_t;
			o->fused_apply = block_difference_operator_mul_fused;
			o->fused_adjoint = block_difference_operator_mul_t_fused;
			o->free = block_difference_operator_data_free;
		}
	}
	return o;
}

#ifdef __cplusplus
}
#endif
