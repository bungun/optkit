#ifndef OPTKIT_OPERATOR_TRANSFORMS_H_
#define OPTKIT_OPERATOR_TRANSFORMS_H_

#include "optkit_abstract_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct transformable_operator {
	operator * o;
	void * (* export)(operator * o);
	void * (* import)(operator * o, void * data);
	ok_status (* abs)(operator * o);
	ok_status (* pow)(operator * o, const ok_float power);
	ok_status (* scale)(operator * o, const ok_float scaling);
	ok_status (* scale_left)(operator * o, const vector * v);
	ok_status (* scale_right)(operator * o, const vector * v);
} transformable_operator;

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_TRANSFORMS_H_ */
