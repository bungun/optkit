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
	memset(op_data, 0, sizeof(*op_data));
	blas_make_handle(&(op_data->sparse_handle));
	sp_make_handle(&(op_data->sparse_handle));
	op_data->A = A;
	return (void *) op_data;
}

ok_status sparse_operator_data_free(void * data)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	OK_CHECK_PTR( op_data );
	OK_RETURNIF_ERR( sp_destroy_handle(op_data->sparse_handle) );
	OK_RETURNIF_ERR( blas_destroy_handle(op_data->sparse_handle) );
	ok_free(op_data);
	return OPTKIT_SUCCESS;
}

ok_status sparse_operator_mul(void * data, vector * input, vector * output)
{
	OK_CHECK_PTR(data);
	return sp_blas_gemv(((sparse_operator_data *) data)->sparse_handle,
		CblasNoTrans, kOne, ((sparse_operator_data *) data)->A, input,
		kZero, output);
}

ok_status sparse_operator_mul_t(void * data, vector * input, vector * output)
{
	OK_CHECK_PTR(data);
	return sp_blas_gemv(((sparse_operator_data *) data)->sparse_handle,
		CblasTrans, kOne, ((sparse_operator_data *) data)->A, input,
		kZero, output);
}

ok_status sparse_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	OK_CHECK_PTR(data);
	return sp_blas_gemv(((sparse_operator_data *) data)->sparse_handle,
		CblasNoTrans, alpha, ((sparse_operator_data *) data)->A, input,
		beta, output);
}

ok_status sparse_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	OK_CHECK_PTR(data);
	return sp_blas_gemv(((sparse_operator_data *) data)->sparse_handle,
		CblasTrans, alpha, ((sparse_operator_data *) data)->A, input,
		beta, output);
}

operator * sparse_operator_alloc(sp_matrix * A)
{
	operator * o = OK_NULL;
		if (A && A->val && A->ind && A->ptr) {
		o = malloc(sizeof(*o));
		memset(o, 0, sizeof(*o));
		o->kind = (A->order == CblasColMajor) ? OkOperatorSparseCSC :
				OkOperatorSparseCSR;
		o->size1 = A->size1;
		o->size2 = A->size2;
		o->data = sparse_operator_data_alloc(A);
		o->apply = sparse_operator_mul;
		o->adjoint = sparse_operator_mul_t;
		o->fused_apply = sparse_operator_mul_fused;
		o->fused_adjoint = sparse_operator_mul_t_fused;
		o->free = sparse_operator_data_free;
	}
	return o;
}

static ok_status sparse_operator_typecheck(operator * A,
	const char * caller)
{
	OK_CHECK_OPERATOR(A);
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

	if (!err)
		return ((sparse_operator_data *) A->data)->A;
	else
		return OK_NULL;
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

void * sparse_operator_import(operator * A, void * data)
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
		data = OK_NULL;
	}

	return data;
}

ok_status sparse_operator_abs(operator * A)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "abs") );
	return sp_matrix_abs(((sparse_operator_data *) A->data)->A);
}

ok_status sparse_operator_pow(operator * A, const ok_float power)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "pow") );
	return sp_matrix_pow(((sparse_operator_data *) A->data)->A, power);
}

ok_status sparse_operator_scale(operator * A, const ok_float scaling)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "scale") );
	return sp_matrix_scale(((sparse_operator_data *) A->data)->A, scaling);
}

ok_status sparse_operator_scale_left(operator * A, const vector * v)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "scale_left") );
	return sp_matrix_scale_left(
		((sparse_operator_data *) A->data)->sparse_handle,
		((sparse_operator_data *) A->data)->A, v);
}

ok_status sparse_operator_scale_right(operator * A, const vector * v)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "scale_right") );
	return sp_matrix_scale_right(
		((sparse_operator_data *) A->data)->sparse_handle,
		((sparse_operator_data *) A->data)->A, v);
}

transformable_operator * sparse_operator_to_transformable(operator * A)
{
	ok_status err = sparse_operator_typecheck(A, "to_transformable");
	transformable_operator * t = OK_NULL;

	if (!err) {
		t = malloc(sizeof(*t));
		t->o = A;
		t->export = sparse_operator_export;
		t->import = sparse_operator_import;
		t->abs = sparse_operator_abs;
		t->pow = sparse_operator_pow;
		t->scale = sparse_operator_scale;
		t->scale_left = sparse_operator_scale_left;
		t->scale_right = sparse_operator_scale_right;
	}

	return t;
}

#ifdef __cplusplus
}
#endif
