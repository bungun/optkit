#ifndef OPTKIT_ABSTRACT_OPERATOR_H_
#define OPTKIT_ABSTRACT_OPERATOR_H_

#include "optkit_defs.h"
#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OK_CHECK_OPERATOR
#define OK_CHECK_OPERATOR(o) \
	do { \
		if (!o || !o->data) \
			return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED ); \
	} while(0)
#endif

typedef enum OPTKIT_OPERATOR{
	OkOperatorNull = 0,
	OkOperatorIdentity = 101,
	OkOperatorNeg = 102,
	OkOperatorAdd = 103,
	OkOperatorCat = 104,
	OkOperatorSplit = 105,
	OkOperatorDense = 201,
	OkOperatorSparseCSR = 301,
	OkOperatorSparseCSC = 302,
	OkOperatorSparseCOO = 303,
	OkOperatorDiagonal = 401,
	OkOperatorBanded = 402,
	OkOperatorTriangular = 403,
	OkOperatorKronecker = 404,
	OkOperatorToeplitz = 405,
	OkOperatorCirculant = 406,
	OkOperatorConvolution = 501,
	OkOperatorCircularConvolution = 502,
	OkOperatorFourier = 503,
	OkOperatorDifference = 504,
	OkOperatorUpsampling = 505,
	OkOperatorDownsampling = 506,
	OkOperatorBlockDifference = 507,
	OkOperatorDirectProjection = 901,
	OkOperatorIndirectProjection = 902,
	OkOperatorOther = 1000
} OPTKIT_OPERATOR;

typedef struct abstract_linear_operator {
	size_t size1, size2;
	void *data;
	ok_status (* apply)(void *data, vector *input, vector *output);
	ok_status (* adjoint)(void *data, vector *input, vector *output);
	ok_status (* fused_apply)(void *data, ok_float alpha, vector *input,
		ok_float beta, vector *output);
	ok_status (* fused_adjoint)(void *data, ok_float alpha, vector *input,
		ok_float beta, vector *output);
	ok_status (* free)(void *data);
	OPTKIT_OPERATOR kind;
} abstract_operator;

static const char * optkit_op2str(OPTKIT_OPERATOR optype)
{
	switch(optype) {
	case OkOperatorNull:
		return "null operator";
	case OkOperatorIdentity:
		return "identity operator";
	case OkOperatorNeg:
		return "negation operator";
	case OkOperatorAdd:
		return "addition operator";
	case OkOperatorCat:
		return "concatenation operator";
	case OkOperatorSplit:
		return "splitting operator";
	case OkOperatorDense:
		return "dense operator";
	case OkOperatorSparseCSC:
		return "sparse CSC operator";
	case OkOperatorSparseCSR:
		return "sparse CSR operator";
	case OkOperatorSparseCOO:
		return "sparse COO operator";
	case OkOperatorDiagonal:
		return "diagonal operator";
	default:
		return "unknown operator";
	}
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_ABSTRACT_OPERATOR_H_ */
