#include "optkit_operator_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sparse_operator_export {
	ok_float * val;
	ok_int * ind, * ptr;
};

void * sparse_operator_data_alloc(sp_matrix * A)
{
	sparse_operator_data * op_data;
	op_data = malloc(sizeof(*op_data));
	blas_make_handle(&(op_data->dense_handle));
	sp_make_handle(&(op_data->sparse_handle));
	op_data->A = A;
	return (void *) op_data;
}

void sparse_operator_data_free(void * data)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_destroy_handle(op_data->sparse_handle);
	blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
}

void sparse_operator_mul(void * data, vector * input, vector * output)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasNoTrans, kOne, op_data->A,
		input, kZero, output);
}

void sparse_operator_mul_t(void * data, vector * input, vector * output)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasTrans, kOne, op_data->A,
		input, kZero, output);
}

void sparse_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasNoTrans, alpha, op_data->A,
		input, beta, output);
}

void sparse_operator_mul_t_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasTrans, alpha, op_data->A,
		input, beta, output);
}

operator * sparse_operator_alloc(sp_matrix * A)
{
	OPTKIT_OPERATOR sparseop_kind = (A->order == CblasColMajor) ?
					OkOperatorSparseCSC :
					OkOperatorSparseCSR;

	return operator_alloc(sparseop_kind, A->size1, A->size2,
		sparse_operator_data_alloc(A),
		sparse_operator_mul, sparse_operator_mul_t,
		sparse_operator_mul_fused, sparse_operator_mul_t_fused,
		sparse_operator_data_free
	);
}

static ok_status sparse_operator_typecheck(operator * A,
	const char * caller)
{
	if (A->kind != OkOperatorSparseCSR && A->kind != OkOperatorSparseCSC) {
		printf("sparse_operator_%s() %s %s\n", caller, "undefined for",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	} else {
		return OPTKIT_SUCCESS;
	}
}


sp_matrix * sparse_operator_get_matrix_pointer(operator * A)
{
	ok_status err = sparse_operator_typecheck(A, "get_matrix_pointer");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		return op_data->A;
	} else {
		return OK_NULL;
	}
}

void * sparse_operator_export(operator * A)
{
	ok_status err = sparse_operator_typecheck(A, "export");
	sparse_operator_data * op_data = OK_NULL;
	struct sparse_operator_export * export = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		export = malloc(sizeof(*export));
		export->val = malloc(op_data->A->nnz * sizeof(ok_float));
		export->ind = malloc(op_data->A->nnz * sizeof(ok_int));
		export->ptr = malloc(op_data->A->ptrlen * sizeof(ok_int));

		sp_matrix_memcpy_am(export->val, export->ind, export->ptr,
			op_data->A);
	}
	return (void *) export;
}

ok_status sparse_operator_import(operator * A, void * data)
{
	ok_status err = sparse_operator_typecheck(A, "import");
	sparse_operator_data * op_data = OK_NULL;
	struct sparse_operator_export * import = OK_NULL;

	if (!err && data) {
		op_data = (sparse_operator_data *) A->data;
		import = (struct sparse_operator_export *) data;
		sp_matrix_memcpy_ma(op_data->sparse_handle, op_data->A,
			import->val, import->ind, import->ptr);

		ok_free(import->val);
		ok_free(import->val);
		ok_free(import->val);
		ok_free(import);
	}

	return err;
}

ok_status sparse_operator_abs(operator * A)
{
	ok_status err = sparse_operator_typecheck(A, "abs");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_abs(op_data->A);
	}
	return err;
}

ok_status sparse_operator_pow(operator * A, const ok_float power)
{
	ok_status err = sparse_operator_typecheck(A, "pow");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_pow(op_data->A, power);
	}
	return err;
}

ok_status sparse_operator_scale(operator * A, const ok_float scaling)
{
	ok_status err = sparse_operator_typecheck(A, "scale");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_scale(op_data->A, scaling);
	}
	return err;
}

ok_status sparse_operator_scale_left(operator * A, const vector * v)
{
	ok_status err = sparse_operator_typecheck(A, "scale_left");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_scale_left(op_data->sparse_handle, op_data->A, v);
	}
	return err;
}

ok_status sparse_operator_scale_right(operator * A, const vector * v)
{
	ok_status err = sparse_operator_typecheck(A, "scale_right");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_scale_right(op_data->sparse_handle, op_data->A, v);
	}
	return err;
}

#ifdef __cplusplus
}
#endif