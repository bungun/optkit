
#include "optkit_upsampling_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline int upsampling_dims_compatible(const int transpose,
	const upsamplingvec * u, const size_t input_dim1,
	const size_t input_dim2, const size_t output_dim1,
	const size_t output_dim2)
{
	if (transpose)
		return (input_dim2 == output_dim2) &&
		((u->size2 <= output_dim1) && (u->size1 == input_dim1));
	else
		return (input_dim2 == output_dim2) &&
		((u->size1 == output_dim1) && (u->size2 <= input_dim1));
}

ok_status upsamplingvec_alloc(upsamplingvec * u, size_t size1, size_t size2)
{
	if (!u)
		return OPTKIT_ERROR_UNALLOCATED;
	else if (u->indices)
		return OPTKIT_ERROR_OVERWRITE;
	u->size1 = size1;
	u->size2 = size2;
	indvector_calloc(&(u->vec), size1);
	u->indices = u->vec.data;
	u->stride = u->vec.stride;
	return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_free(upsamplingvec * u)
{
	if (!u || !(u->indices))
		return OPTKIT_ERROR_UNALLOCATED;
	ok_free(u->vec.data);
	u->vec.size = 0;
	u->vec.stride = 0;
	u->indices = OK_NULL;
	u->size1 = 0;
	u->size2 = 0;
	u->stride = 0;
	return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_check_bounds(const upsamplingvec * u)
{
	if (indvector_max(&(u->vec)) < u->size2)
		return OPTKIT_SUCCESS;
	else
		return OPTKIT_ERROR_DIMENSION_MISMATCH;
}

ok_status upsamplingvec_subvector(upsamplingvec * usub, upsamplingvec * u,
	size_t offset1, size_t offset2, size_t length1, size_t length2)
{
	indvector_subvector(&(usub->vec), &(u->vec), offset1, length1);
	usub->indices = usub->vec.data;
	usub->size1 = length1;
	usub->size2 = length2 - offset2;
	return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_mul_matrix(const enum CBLAS_TRANSPOSE transU,
	const enum CBLAS_TRANSPOSE transI, const enum CBLAS_TRANSPOSE transO,
	const ok_float alpha, upsamplingvec * u, matrix * M_in, ok_float beta,
	matrix * M_out)
{
	size_t i, dim_in1, dim_in2, dim_out1, dim_out2;
	size_t ptr_stride_in, ptr_stride_out;
	int stride_in, stride_out;
	int transpose = transU == CblasTrans;

	if ((!u || !M_in || !M_out) ||
	    (!u->indices || !M_in->data ||!M_out->data))
		return OPTKIT_ERROR_UNALLOCATED;

	dim_in1 = (transI == CblasNoTrans) ? M_in->size1 : M_in->size2;
	dim_in2 = (transI == CblasNoTrans) ? M_in->size2 : M_in->size1;
	dim_out1 = (transO == CblasNoTrans) ? M_out->size1 : M_out->size2;
	dim_out2 = (transO == CblasNoTrans) ? M_out->size2 : M_out->size1;

	if (!upsampling_dims_compatible(transpose, u, dim_in1, dim_in2,
		dim_out1, dim_out2))
		return OPTKIT_ERROR_DIMENSION_MISMATCH;

	stride_in =
		((transI == CblasNoTrans) == (M_in->order == CblasRowMajor)) ?
		1 : (int) M_in->ld;
	stride_out =
		((transO == CblasNoTrans) == (M_out->order == CblasRowMajor)) ?
		1 : (int) M_out->ld;
	ptr_stride_in = (stride_in == 1) ? M_in->ld : 1;
	ptr_stride_out = (stride_out == 1) ? M_out->ld : 1;

	matrix_scale(M_out, beta);

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

ok_status upsamplingvec_count(const upsamplingvec * u, vector * counts)
{
	size_t i;
	if ((!u || !counts) || (!u->indices || !counts->data))
		return OPTKIT_ERROR_UNALLOCATED;

	if (u->size2 > counts->size)
		return OPTKIT_ERROR_DIMENSION_MISMATCH;

	vector_scale(counts, kZero);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < u->size1; ++i)
		counts->data[u->indices[i * u->stride] * counts->stride] +=
			kOne;

	return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_shift(upsamplingvec * u, const size_t shift,
	const enum OPTKIT_TRANSFORM direction)
{
	size_t i;
	if (direction == OkTransformDecrement) {
		if (shift >= u->size2)
			return OPTKIT_ERROR_OUT_OF_BOUNDS;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < u->size1; ++i)
			u->indices[i * u->stride] -= shift;
	} else {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < u->size1; ++i)
			u->indices[i * u->stride] += shift;
	}

	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
