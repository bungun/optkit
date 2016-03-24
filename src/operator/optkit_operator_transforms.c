#include "optkit_operator_transforms.h"

#ifdef __cplusplus
extern "C" {
#endif

optkit_status operator_copy(operator * A, void * data,
	enum OPTKIT_COPY_DIRECTION dir)
{
	if (operator->kind == OkOperatorDense) {
		return dense_operator_copy(A, data, dir);
	} else if (operator->kind == OkOperatorSparseCSR ||
		   operator->kind == OkOperatorSparseCSC) {
		return sparse_operator_copy(A, data, dir);
	} else {
		printf("%s\n", "operator_copy() undefined for %s",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}

optkit_status operator_abs(operator * A)
{
	if (operator->kind == OkOperatorDense) {
		return dense_operator_abs(A);
	} else if (operator->kind == OkOperatorSparseCSR ||
		   operator->kind == OkOperatorSparseCSC) {
		return sparse_operator_abs(A);
	} else {
		printf("%s\n", "operator_abs() undefined for %s",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}

optkit_status operator_pow(operator * A, ok_float power)
{
	if (operator->kind == OkOperatorDense) {
		return dense_operator_pow(A, power);
	} else if (operator->kind == OkOperatorSparseCSR ||
		   operator->kind == OkOperatorSparseCSC) {
		return sparse_operator_pow(A, power);
	} else {
		printf("%s\n", "operator_pow() undefined for %s",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}

optkit_status operator_scale(operator * A, ok_float scaling)
{
	if (operator->kind == OkOperatorDense) {
		return dense_operator_scale(A, scaling);
	} else if (operator->kind == OkOperatorSparseCSR ||
		   operator->kind == OkOperatorSparseCSC) {
		return sparse_operator_scale(A, scaling);
	} else {
		printf("%s\n", "operator_scale() undefined for %s",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}

optkit_status operator_scale_left(operator * A, vector * v)
{
	if (v->size != A->size1) {
		printf("%s\n", "dimensions of 'v' incompatible with 'A'");
		printf("rows A %u\n", (uint) A->size1);
		printf("size v %s\n", (uint) v->size);
		return OPTKIT_ERROR;
	}
	if (operator->kind == OkOperatorDense) {
		return dense_operator_scale_left(A, scaling);
	} else if (operator->kind == OkOperatorSparseCSR ||
		   operator->kind == OkOperatorSparseCSC) {
		return sparse_operator_scale_left(A, scaling);
	} else {
		printf("%s\n", "operator_scale_left() undefined for %s",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}

optkit_status operator_scale_right(operator * A, vector * v)
{
	if (v->size != A->size2) {
		printf("%s\n", "dimensions of 'v' incompatible with 'A'");
		printf("cols A %u\n", (uint) A->size2);
		printf("size v %s\n", (uint) v->size);
		return OPTKIT_ERROR;
	}

	if (operator->kind == OkOperatorDense) {
		return dense_operator_scale_left(A, scaling);
	} else if (operator->kind == OkOperatorSparseCSR ||
		   operator->kind == OkOperatorSparseCSC) {
		return sparse_operator_scale_right(A, v);
	} else {
		printf("%s\n", "operator_scale_right() undefined for %s",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	}
}


#ifdef __cplusplus
}
#endif