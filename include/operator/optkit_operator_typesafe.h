#ifndef OPTKIT_OPERATOR_TYPESAFE_H_
#define OPTKIT_OPERATOR_TYPESAFE_H_

#include "optkit_abstract_operator.h"
#include "optkit_operator_transforms.h"

#ifdef __cplusplus
extern "C" {
#endif

/* STUB */
ok_status typesafe_operator_scale(abstract_operator * A, const ok_float scaling)
{
	return OPTKIT_ERROR;
}

/* STUB */
ok_status typesafe_operator_scale_left(abstract_operator * A, const vector * v)
{
	return OPTKIT_ERROR;

}

/* STUB */
ok_status typesafe_operator_scale_right(abstract_operator * A, const vector * v)
{
	return OPTKIT_ERROR;

}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_TYPESAFE_H_ */
