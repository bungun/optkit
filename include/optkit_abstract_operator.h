#ifndef OPTKIT_ABSTRACT_OPERATOR_H_
#define OPTKIT_ABSTRACT_OPERATOR_H_

#include "optkit_defs.h"

 __cplusplus
extern "C" {
#endif

typedef enum OPERATOR{
	Null_Operator = 0,
	Id_Operator = 101,
	Neg_Operator = 102,
	Add_Operator = 103,
	Cat_Operator = 104,
	Split_Operator = 105,
	Dense_Operator = 201,
	SparseCSR_Operator = 301,
	SparseCSC_Operator = 302,
	SparseCOO_Operator = 303,
	Diagonal_Operator = 401,
	Banded_Operator = 402,
	Triangular_Operator = 403,
	Kronecker_Operator = 404,
	Toeplitz_Operator = 405,
	Circulant_Operator = 406,
	Convolution_Operator = 501,
	CircularConvolution_Operator = 502,
	Fourier_Operator = 503,
	Difference_Operator = 504,
	Upsampling_Operator = 505,
	Downsampling_Operator = 506,
	BlockDifference_Operator = 507,
	DirectProjection_Operator = 901,
	IndirectProjection_Operator = 902,
	Other_Operator = 1000
} OPTKIT_OPERATOR;

typedef struct abstract_linear_operator{
	size_t size1, size2;
	void * data;
	void (* apply)(void * data, vector * input, vector * output);
	void (* adjoint)(void * data, vector * input, vector * output);
	void (* fused_apply)(void * data, ok_float alpha, vector * input,
		ok_float beta, vector * output);
	void (* fused_adjoint)(void * data, ok_float alpha, vector * input,
		ok_float beta, vector * output);
	void (* free)(void * data);
	OPTKIT_OPERATOR kind;
} operator;


operator * operator_alloc(OPTKIT_OPERATOR kind,
	size_t size1, size_t size2, void * data,
	void (* apply), void (* adjoint),
	void (* fused_apply), void (* fused_adjoint),
	void (* free)){

	operator * op;
	op = malloc(sizeof(*op));
	op->size1 = size1;
	op->size2 = size2;
	op->apply = apply;
	op->adjoint = adjoint;
	op->fused_apply = op->fused_apply;
	op->fused_adjoint = op->fused_adjoint;
	op->free = op->free;
	op->kind = kind;
	return op;
}

/* GENERIC OPERATOR FREE */
void operator_free(operator * op){
	if (op->free && op->data) op->free(op->data);
	ok_free(op);
}


#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_ABSTRACT_OPERATOR_H_ */