#ifndef OPTKIT_PROJECTOR_H_
#define OPTKIT_PROJECTOR_H_

#include "optkit_dense.h"
#include "optkit_abstract_operator.h"
#include "optkit_cg.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OK_CHECK_PROJECTOR
#define OK_CHECK_PROJECTOR(P) \
	do { \
		if (!P || !P->data) \
			return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED ); \
	} while(0)
#endif

typedef enum OPTKIT_PROJECTOR {
	OkProjectorDenseDirect = 101,
	OkProjectorSparseDirect = 102,
	OkProjectorIndirect = 103
} OPTKIT_PROJECTOR;

typedef struct projector {
	OPTKIT_PROJECTOR kind;
	size_t size1, size2;
	void * data;
	ok_status (* initialize)(void * data, const int normalize);
	ok_status (* project)(void * data, vector * x_in, vector * y_in,
		vector * x_out, vector * y_out, ok_float tol);
	ok_status (* free)(void * data);
} projector;

ok_status projector_normalization(projector * P, int * normalized);
ok_status projector_get_norm(projector * P, ok_float * norm);

typedef struct direct_projector {
	matrix * A;
	matrix * L;
	ok_float normA;
	int skinny, normalized;
} direct_projector;

ok_status direct_projector_alloc(direct_projector * P, matrix * A);
ok_status direct_projector_initialize(void * linalg_handle,
	direct_projector * P, const int normalize);
ok_status direct_projector_project(void * linalg_handle, direct_projector * P,
	vector * x_in, vector * y_in, vector * x_out, vector * y_out);
ok_status direct_projector_free(direct_projector * P);

typedef struct indirect_projector {
	abstract_operator * A;
	void * cgls_work;
	uint flag;
} indirect_projector;

ok_status indirect_projector_alloc(indirect_projector * P, abstract_operator * A);
ok_status indirect_projector_initialize(void * linalg_handle,
	indirect_projector * P, const int normalize);
ok_status indirect_projector_project(void * linalg_handle,
	indirect_projector * P, vector * x_in, vector * y_in, vector * x_out,
	vector * y_out);
ok_status indirect_projector_free(indirect_projector * P);

typedef struct dense_direct_projector {
	matrix * A;
	matrix * L;
	void * linalg_handle;
	ok_float normA;
	int skinny, normalized;
} dense_direct_projector;

void * dense_direct_projector_data_alloc(matrix * A);
ok_status dense_direct_projector_data_free(void * data);
ok_status dense_direct_projector_initialize(void * data, const int normalize);
ok_status dense_direct_projector_project(void * data, vector * x_in,
	vector * y_in, vector * x_out, vector * y_out, ok_float tol);
projector * dense_direct_projector_alloc(matrix * A);

typedef struct indirect_projector_generic {
	abstract_operator * A;
	void * cgls_work;
	void * linalg_handle;
	ok_float normA;
	int normalized;
	uint flag;
} indirect_projector_generic;

void * indirect_projector_data_alloc(abstract_operator * A);
ok_status indirect_projector_data_free(void * data);
ok_status indirect_projector_g_initialize(void * data, const int normalize);
ok_status indirect_projector_g_project(void * data, vector * x_in,
	vector * y_in, vector * x_out, vector * y_out, ok_float tol);
projector * indirect_projector_generic_alloc(abstract_operator * A);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_PROJECTOR_H_ */
