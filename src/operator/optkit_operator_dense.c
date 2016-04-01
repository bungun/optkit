#include "optkit_operator_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DENSE LINEAR OPERATOR */
void * dense_operator_data_alloc(matrix * A)
{
	dense_operator_data * op_data;
	op_data = malloc(sizeof(*op_data));
	blas_make_handle(&(op_data->dense_handle));
	op_data->A = A;
	return (void *) op_data;
}

void dense_operator_data_free(void * data)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
}

void dense_operator_mul(void * data, vector * input, vector * output)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasNoTrans, kOne, op_data->A, input,
		kZero, output);
}

void dense_operator_mul_t(void * data, vector * input, vector * output)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasTrans, kOne, op_data->A, input,
		kZero, output);
}

void dense_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasNoTrans, alpha, op_data->A, input,
		beta, output);
}

void dense_operator_mul_t_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasTrans, alpha, op_data->A, input,
		beta, output);
}

operator * dense_operator_alloc(matrix * A)
{
	operator * o = OK_NULL;
	o = malloc(sizeof(*o));
	o->kind = OkOperatorDense;
	o->size1 = A->size1;
	o->size2 = A->size2;
	o->data = dense_operator_data_alloc(A);
	o->apply = dense_operator_mul;
	o->adjoint = dense_operator_mul_t;
	o->fused_apply = dense_operator_mul_fused;
	o->fused_adjoint = dense_operator_mul_t_fused;
	o->free = dense_operator_data_free;
	return o;
}

static ok_status dense_operator_typecheck(operator * A, const char * caller)
{
	if (A->kind != OkOperatorDense) {
		printf("dense_operator_%s() %s %s\n", caller, "undefined for",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	} else {
		return OPTKIT_SUCCESS;
	}
}

matrix * dense_operator_get_matrix_pointer(operator * A)
{
	ok_status err = dense_operator_typecheck(A, "get_matrix_pointer");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		return op_data->A;
	} else {
		return OK_NULL;
	}
}


void * dense_operator_export(operator * A)
{
	ok_status err = dense_operator_typecheck(A, "export");
	dense_operator_data * op_data = OK_NULL;
	ok_float * export = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		export = malloc(op_data->A->size1 * op_data->A->size2 *
			sizeof(*export));
		matrix_memcpy_am(export, op_data->A, op_data->A->order);
	}
	return (void *) export;
}

void * dense_operator_import(operator * A, void * data)
{
	ok_status err = dense_operator_typecheck(A, "import");
	dense_operator_data * op_data = OK_NULL;
	ok_float * import = OK_NULL;

	if (!err && data) {
		op_data = (dense_operator_data *) A->data;
		import = (ok_float *) data;
		matrix_memcpy_ma(op_data->A, import, op_data->A->order);
		ok_free(import);
		data = OK_NULL;
	}

	return data;
}

ok_status dense_operator_abs(operator * A)
{
	ok_status err = dense_operator_typecheck(A, "abs");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_abs(op_data->A);
	}
	return err;
}

ok_status dense_operator_pow(operator * A, const ok_float power)
{
	ok_status err = dense_operator_typecheck(A, "pow");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_pow(op_data->A, power);
	}
	return err;
}

ok_status dense_operator_scale(operator * A, const ok_float scaling)
{
	ok_status err = dense_operator_typecheck(A, "scale");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_scale(op_data->A, scaling);
	}
	return err;
}

ok_status dense_operator_scale_left(operator * A, const vector * v)
{
	ok_status err = dense_operator_typecheck(A, "scale_left");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_scale_left(op_data->A, v);
	}
	return err;
}

ok_status dense_operator_scale_right(operator * A, const vector * v)
{
	ok_status err = dense_operator_typecheck(A, "scale_right");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_scale_right(op_data->A, v);
	}
	return err;
}

transformable_operator * dense_operator_to_transformable(operator * A)
{
	ok_status err = dense_operator_typecheck(A, "to_transformable");
	transformable_operator * t = OK_NULL;

	if (!err) {
		t = malloc(sizeof(*t));
		t->o = A;
		t->export = dense_operator_export;
		t->import = dense_operator_import;
		t->abs = dense_operator_abs;
		t->pow = dense_operator_pow;
		t->scale = dense_operator_scale;
		t->scale_left = dense_operator_scale_left;
		t->scale_right = dense_operator_scale_right;
	}

	return t;
}

#ifdef __cplusplus
}
#endif
