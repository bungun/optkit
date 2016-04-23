#ifndef OPTKIT_PROJECTOR_H_
#define OPTKIT_PROJECTOR_H_

#include "optkit_dense.h"
#include "optkit_abstract_operator.h"
#include "optkit_cg.h"

#ifdef __cplusplus
extern "C" {
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
	void (* initialize)(void * data, const int normalize);
	void (* project)(void * data, vector * x_in, vector * y_in,
		vector * x_out, vector * y_out, ok_float tol);
	void (* free)(void * data);
} projector;

int projector_normalization(projector * P);
ok_float projector_get_norm(projector * P);

typedef struct DirectProjector {
	matrix * A;
	matrix * L;
	ok_float normA;
	int skinny, normalized;
} direct_projector;

void direct_projector_alloc(direct_projector * P, matrix * A);
void direct_projector_initialize(void * linalg_handle, direct_projector * P,
	const int normalize);
void direct_projector_project(void * linalg_handle, direct_projector * P,
	vector * x_in, vector * y_in, vector * x_out, vector * y_out);
void direct_projector_free(direct_projector * P);

typedef struct IndirectProjector {
	operator * A;
	void * cgls_work;
} indirect_projector;

void indirect_projector_alloc(indirect_projector * P, operator * A);
void indirect_projector_initialize(void * linalg_handle, indirect_projector * P,
	const int normalize);
void indirect_projector_project(void * linalg_handle, indirect_projector * P,
	vector * x_in, vector * y_in, vector * x_out, vector * y_out);
void indirect_projector_free(indirect_projector * P);

typedef struct dense_direct_projector {
	matrix * A;
	matrix * L;
	void * linalg_handle;
	ok_float normA;
	int skinny, normalized;
} dense_direct_projector;

void * dense_direct_projector_data_alloc(matrix * A);
void dense_direct_projector_data_free(void * data);
void dense_direct_projector_initialize(void * data, const int normalize);
void dense_direct_projector_project(void * data, vector * x_in, vector * y_in,
	vector * x_out, vector * y_out, ok_float tol);
projector * dense_direct_projector_alloc(matrix * A);

typedef struct indirect_projector_generic {
	operator * A;
	void * cgls_work;
	void * linalg_handle;
	ok_float normA;
	int normalized;
} indirect_projector_generic;

void * indirect_projector_data_alloc(operator * A);
void indirect_projector_data_free(void * data);
void indirect_projector_g_initialize(void * data, const int normalize);
void indirect_projector_g_project(void * data, vector * x_in, vector * y_in,
	vector * x_out, vector * y_out, ok_float tol);
projector * indirect_projector_generic_alloc(operator * A);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_PROJECTOR_H_ */
