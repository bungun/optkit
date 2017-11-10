#ifndef OPTKIT_OPERATOR_TRANSFORMS_H_
#define OPTKIT_OPERATOR_TRANSFORMS_H_

#include "optkit_abstract_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct transformable_operator {
	abstract_operator * o;
	void * (* export)(abstract_operator * o);
	void * (* import)(abstract_operator * o, void * data);
	ok_status (* abs)(abstract_operator * o);
	ok_status (* pow)(abstract_operator * o, const ok_float power);
	ok_status (* scale)(abstract_operator * o, const ok_float scaling);
	ok_status (* scale_left)(abstract_operator * o, const vector * v);
	ok_status (* scale_right)(abstract_operator * o, const vector * v);
} transformable_operator;

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_TRANSFORMS_H_ */
