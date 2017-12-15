#include "optkit_upsampling_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status upsamplingvec_mul_matrix(void *linalg_handle,
	const enum CBLAS_TRANSPOSE transU, const enum CBLAS_TRANSPOSE transI,
	const enum CBLAS_TRANSPOSE transO, const ok_float alpha,
	upsamplingvec *u, matrix *M_in, ok_float beta, matrix *M_out)
{
	OK_CHECK_UPSAMPLINGVEC(u);
	OK_CHECK_MATRIX(M_in);
	OK_CHECK_MATRIX(M_out);

	size_t i, dim_in1, dim_in2, dim_out1, dim_out2;
	size_t ptr_stride_in, ptr_stride_out;
	int stride_in, stride_out;
	const int transpose = transU == CblasTrans;

	if ((!u || !M_in || !M_out) ||
	    (!u->indices || !M_in->data ||!M_out->data))
		return OPTKIT_ERROR_UNALLOCATED;

	dim_in1 = (transI == CblasNoTrans) ? M_in->size1 : M_in->size2;
	dim_in2 = (transI == CblasNoTrans) ? M_in->size2 : M_in->size1;
	dim_out1 = (transO == CblasNoTrans) ? M_out->size1 : M_out->size2;
	dim_out2 = (transO == CblasNoTrans) ? M_out->size2 : M_out->size1;

	if (!upsampling_dims_compatible(transpose, u, dim_in1, dim_in2,
		dim_out1, dim_out2))
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	stride_in =
		((transI == CblasNoTrans) == (M_in->order == CblasRowMajor)) ?
		1 : (int) M_in->ld;
	stride_out =
		((transO == CblasNoTrans) == (M_out->order == CblasRowMajor)) ?
		1 : (int) M_out->ld;
	ptr_stride_in = (stride_in == 1) ? M_in->ld : 1;
	ptr_stride_out = (stride_out == 1) ? M_out->ld : 1;

	OK_RETURNIF_ERR( matrix_scale(M_out, beta) );

	if (!transpose)
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < dim_out1; ++i)
			CBLAS(axpy)((int) dim_in2, alpha,
				M_in->data + u->indices[i*u->stride] *
				ptr_stride_in,
				stride_in, M_out->data + i * ptr_stride_out,
				stride_out);
	else
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < dim_in1; ++i)
			CBLAS(axpy)((int) dim_in2, alpha,
				M_in->data + i * ptr_stride_in, stride_in,
				M_out->data + u->indices[i*u->stride] *
				ptr_stride_out,
				stride_out);

	return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_count(const upsamplingvec *u, vector *counts)
{
	size_t i;
	if ((!u || !counts) || (!u->indices || !counts->data))
		return OPTKIT_ERROR_UNALLOCATED;

	if (u->size2 > counts->size)
		return OPTKIT_ERROR_DIMENSION_MISMATCH;

	OK_RETURNIF_ERR( vector_scale(counts, kZero) );

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < u->size1; ++i)
		counts->data[u->indices[i * u->stride] * counts->stride] +=
			kOne;

	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
