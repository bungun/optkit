#ifndef OPTKIT_PROJECTOR_H_
#define OPTKIT_PROJECTOR_H_

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

void projectorlib_version(int * maj, int * min, int * change, int * status);


typedef struct DirectProjector {
	matrix * A;
	matrix * L;
	ok_float normA;
	int skinny, normalized;
} direct_projector;


void direct_projector_alloc(direct_projector * P, matrix * A);
void direct_projector_initialize(void * linalg_handle, direct_projector * P,
	int normalize);
void direct_projector_project(void * linalg_handle, direct_projector * P,
	vector * x_in, vector * y_in, vector * x_out, vector * y_out);
void direct_projector_free(direct_projector * P);

typedef struct IndirectProjector {
	matrix * A;
} indirect_projector;

void indirect_projector_alloc(direct_projector * P, matrix * A);
void indirect_projector_initialize(void * linalg_handle, direct_projector * P,
	int normalize);
void indirect_projector_project(void * linalg_handle, direct_projector * P,
	vector * x_in, vector * y_in, vector * x_out, vector * y_out);
void indirect_projector_free(direct_projector * P);


#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_PROJECTOR_H_ */