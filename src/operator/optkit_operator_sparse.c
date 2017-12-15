#include "optkit_operator_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sparse_operator_exporter {
	ok_float *val;
	ok_int *ind, *ptr;
};

void * sparse_operator_data_alloc(sp_matrix *A)
{
	ok_status err = OPTKIT_SUCCESS;
	sparse_operator_data *op_data = OK_NULL;

	if (!A || !A->val || !A->ind | !A->ptr)
		err = OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	if (!err) {
		ok_alloc(op_data, sizeof(*op_data));
		OK_CHECK_ERR( err,
			sp_make_handle(&(op_data->sparse_handle)) );
		op_data->A = A;
		if (err) {
			sparse_operator_data_free(op_data);
			op_data = OK_NULL;
		}
	}
	return (void *) op_data;
}

ok_status sparse_operator_data_free(void *data)
{
	sparse_operator_data *op_data = (sparse_operator_data *) data;
	OK_CHECK_PTR( op_data );
	OK_RETURNIF_ERR( sp_destroy_handle(op_data->sparse_handle) );
	ok_free(op_data);
	return OPTKIT_SUCCESS;
}

ok_status sparse_operator_mul(void *data, vector *input, vector *output)
{
	OK_CHECK_PTR(data);
	return sp_blas_gemv(((sparse_operator_data *) data)->sparse_handle,
		CblasNoTrans, kOne, ((sparse_operator_data *) data)->A, input,
		kZero, output);
}

ok_status sparse_operator_mul_t(void *data, vector *input, vector *output)
{
	OK_CHECK_PTR(data);
	return sp_blas_gemv(((sparse_operator_data *) data)->sparse_handle,
		CblasTrans, kOne, ((sparse_operator_data *) data)->A, input,
		kZero, output);
}

ok_status sparse_operator_mul_fused(void *data, ok_float alpha, vector *input,
	ok_float beta, vector *output)
{
	OK_CHECK_PTR(data);
	return sp_blas_gemv(((sparse_operator_data *) data)->sparse_handle,
		CblasNoTrans, alpha, ((sparse_operator_data *) data)->A, input,
		beta, output);
}

ok_status sparse_operator_mul_t_fused(void *data, ok_float alpha,
	vector *input, ok_float beta, vector *output)
{
	OK_CHECK_PTR(data);
	return sp_blas_gemv(((sparse_operator_data *) data)->sparse_handle,
		CblasTrans, alpha, ((sparse_operator_data *) data)->A, input,
		beta, output);
}

abstract_operator * sparse_operator_alloc(sp_matrix *A)
{
	abstract_operator *o = OK_NULL;
	void *data = OK_NULL;
	if (A && A->val && A->ind && A->ptr) {
		data = sparse_operator_data_alloc(A);
		if (data) {
			ok_alloc(o, sizeof(*o));
			o->kind = (A->order == CblasColMajor) ?
				  OkOperatorSparseCSC : OkOperatorSparseCSR;
			o->size1 = A->size1;
			o->size2 = A->size2;
			o->data = data;
			o->apply = sparse_operator_mul;
			o->adjoint = sparse_operator_mul_t;
			o->fused_apply = sparse_operator_mul_fused;
			o->fused_adjoint = sparse_operator_mul_t_fused;
			o->free = sparse_operator_data_free;
		}
	}
	return o;
}

static ok_status sparse_operator_typecheck(abstract_operator *A,
	const char *caller)
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


sp_matrix * sparse_operator_get_matrix_pointer(abstract_operator *A)
{
	ok_status err = sparse_operator_typecheck(A, "get_matrix_pointer");

	if (!err)
		return ((sparse_operator_data *) A->data)->A;
	else
		return OK_NULL;
}

void * sparse_operator_export(abstract_operator *A)
{
	ok_status err = sparse_operator_typecheck(A, "export");
	sparse_operator_data *op_data = OK_NULL;
	struct sparse_operator_exporter *export = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		ok_alloc(export, sizeof(*export));
		ok_alloc(export->val, op_data->A->nnz * sizeof(*export->val));
		ok_alloc(export->ind, op_data->A->nnz * sizeof(*export->ind));
		ok_alloc(export->ptr, op_data->A->ptrlen * sizeof(*export->ptr));
		sp_matrix_memcpy_am(export->val, export->ind, export->ptr,
			op_data->A);
	}
	return (void *) export;
}

void * sparse_operator_import(abstract_operator *A, void *data)
{
	ok_status err = sparse_operator_typecheck(A, "import");
	sparse_operator_data *op_data = OK_NULL;
	struct sparse_operator_exporter *import = OK_NULL;

	if (!err && data) {
		op_data = (sparse_operator_data *) A->data;
		import = (struct sparse_operator_exporter *) data;
		sp_matrix_memcpy_ma(op_data->sparse_handle, op_data->A,
			import->val, import->ind, import->ptr);

		ok_free(import->val);
		ok_free(import->ind);
		ok_free(import->ptr);
		ok_free(import);
		data = OK_NULL;
	}

	return data;
}

ok_status sparse_operator_abs(abstract_operator *A)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "abs") );
	return sp_matrix_abs(((sparse_operator_data *) A->data)->A);
}

ok_status sparse_operator_pow(abstract_operator *A, const ok_float power)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "pow") );
	return sp_matrix_pow(((sparse_operator_data *) A->data)->A, power);
}

ok_status sparse_operator_scale(abstract_operator *A, const ok_float scaling)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "scale") );
	return sp_matrix_scale(((sparse_operator_data *) A->data)->A, scaling);
}

ok_status sparse_operator_scale_left(abstract_operator *A, const vector *v)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "scale_left") );
	return sp_matrix_scale_left(
		((sparse_operator_data *) A->data)->sparse_handle,
		((sparse_operator_data *) A->data)->A, v);
}

ok_status sparse_operator_scale_right(abstract_operator *A, const vector *v)
{
	OK_RETURNIF_ERR( sparse_operator_typecheck(A, "scale_right") );
	return sp_matrix_scale_right(
		((sparse_operator_data *) A->data)->sparse_handle,
		((sparse_operator_data *) A->data)->A, v);
}

transformable_operator * sparse_operator_to_transformable(abstract_operator *A)
{
	ok_status err = sparse_operator_typecheck(A, "to_transformable");
	transformable_operator *t = OK_NULL;

	if (!err) {
		ok_alloc(t, sizeof(*t));
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
