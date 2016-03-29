#include "optkit_operator_transforms.h"

#ifdef __cplusplus
extern "C" {
#endif

void * operator_export(operator * A)
{
	if (A->kind == OkOperatorDense) {
		return dense_operator_export(A);
	} else if (A->kind == OkOperatorSparseCSR ||
		   A->kind == OkOperatorSparseCSC) {
		return sparse_operator_export(A);
	} else {
		printf("operator_export() undefined for %s\n",
			optkit_op2str(A->kind));
		return OK_NULL;
	}
}

ok_status operator_import(operator * A, void * opdata)
{
	if (A->kind == OkOperatorDense) {
		return dense_operator_import(A, opdata);
	} else if (A->kind == OkOperatorSparseCSR ||
		   A->kind == OkOperatorSparseCSC) {
		return sparse_operator_import(A, opdata);
	} else {
		printf("operator_import() undefined for %s\n",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}


ok_status operator_abs(operator * A)
{
	if (A->kind == OkOperatorDense) {
		return dense_operator_abs(A);
	} else if (A->kind == OkOperatorSparseCSR ||
		   A->kind == OkOperatorSparseCSC) {
		return sparse_operator_abs(A);
	} else {
		printf("operator_abs() undefined for %s\n",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}

ok_status operator_pow(operator * A, ok_float power)
{
	if (A->kind == OkOperatorDense) {
		return dense_operator_pow(A, power);
	} else if (A->kind == OkOperatorSparseCSR ||
		   A->kind == OkOperatorSparseCSC) {
		return sparse_operator_pow(A, power);
	} else {
		printf("operator_pow() undefined for %s\n",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}

ok_status operator_scale(operator * A, ok_float scaling)
{
	if (A->kind == OkOperatorDense) {
		return dense_operator_scale(A, scaling);
	} else if (A->kind == OkOperatorSparseCSR ||
		   A->kind == OkOperatorSparseCSC) {
		return sparse_operator_scale(A, scaling);
	} else {
		printf("operator_scale() undefined for %s\n",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}

ok_status operator_scale_left(operator * A, const vector * v)
{
	if (v->size != A->size1) {
		printf("%s\n", "dimensions of 'v' incompatible with 'A'");
		printf("rows A %u\n", (uint) A->size1);
		printf("size v %u\n", (uint) v->size);
		return OPTKIT_ERROR;
	}
	if (A->kind == OkOperatorDense) {
		return dense_operator_scale_left(A, v);
	} else if (A->kind == OkOperatorSparseCSR ||
		   A->kind == OkOperatorSparseCSC) {
		return sparse_operator_scale_left(A, v);
	} else {
		printf("operator_scale_left() undefined for %s\n",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}

ok_status operator_scale_right(operator * A, const vector * v)
{
	if (v->size != A->size2) {
		printf("%s\n", "dimensions of 'v' incompatible with 'A'");
		printf("cols A %u\n", (uint) A->size2);
		printf("size v %u\n", (uint) v->size);
		return OPTKIT_ERROR;
	}

	if (A->kind == OkOperatorDense) {
		return dense_operator_scale_left(A, v);
	} else if (A->kind == OkOperatorSparseCSR ||
		   A->kind == OkOperatorSparseCSC) {
		return sparse_operator_scale_right(A, v);
	} else {
		printf("operator_scale_right() undefined for %s\n",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}


#ifdef __cplusplus
}
#endif