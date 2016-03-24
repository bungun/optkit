#ifndef OPTKIT_OPERATOR_TRANSFORMS_H_
#define OPTKIT_OPERATOR_TRANSFORMS_H_

#include "optkit_abstract_operator.h"
#include "optkit_operator_dense.h"
#include "optkit_operator_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status operator_copy(operator * A, void * data, OPTKIT_COPY_DIRECTION dir);
ok_status operator_abs(operator * A);
ok_status operator_pow(operator * A, const ok_float power);
ok_status operator_scale(operator * A, const ok_float scaling);
ok_status operator_scale_left(operator * A, const vector * v);
ok_status operator_scale_right(operator * A, const vector * v);

// ok_status operator_pack();
// ok_status operator_unpack();

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_OPERATOR_TRANSFORMS_H_ */