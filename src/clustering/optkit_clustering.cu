#include "optkit_defs_gpu.h"
#include "optkit_clustering.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * given tentative cluster assigments {(i, k)} stored in h, reassign
 * vector i to cluster k (no action if vector i assigned to cluster
 * k already).
 *
 * tally the number of reassignments.
 */
ok_status assign_clusters_l2(matrix * A, matrix * C, upsamplingvec * a2c,
	cluster_aid * h)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t i;
	upsamplingvec * a2c_new = &h->a2c_tentative;

	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	OK_CHECK_UPSAMPLINGVEC(a2c);
	OK_CHECK_PTR(h);
	if (A->size1 != a2c->size1 || a2c->size2 > C->size1 ||
		A->size2 != C->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	h->reassigned = 0;
	for (i = 0; i < A->size1 && !err; ++i) {
		size_t assignment_old, assignment_new;
		cudaStream_t s;
		OK_CHECK_CUDA( err,
			cudaStreamCreate(&s) );
		OK_CHECK_CUDA( err,
			cudaMemcpyAsync(&assignment_old,
				a2c->indices + i * a2c->stride,
				sizeof(assignment_old), cudaMemcpyDeviceToHost,
				s) );
		OK_CHECK_CUDA( err,
			cudaMemcpyAsync(&assignment_new,
				a2c_new->indices + i * a2c_new->stride,
				sizeof(assignment_new), cudaMemcpyDeviceToHost,
				s) );
		if (assignment_old != assignment_new) {
			OK_CHECK_CUDA( err,
				cudaMemcpyAsync(a2c->indices + i *
					a2c->stride, &assignment_new,
					sizeof(ok_float),
					cudaMemcpyHostToDevice, s) );
			++h->reassigned;
		}
		OK_CHECK_CUDA( err,
			cudaStreamDestroy(s) );

	}
	cudaDeviceSynchronize();
	OK_MAX_ERR( err, OK_STATUS_CUDA );
	return err;
}

/*
 * given tentative cluster assigments {(i, k)} stored in h, reassign
 * vector i to cluster k if
 *
 * 	||a_i - c_k||_\infty <= maxdist
 *
 * (no action if vector i assigned to cluster k already)
 *
 * tally the number of reassignments.
 *
 */


ok_status assign_inner_loop(matrix * A, matrix * C, upsamplingvec * a2c,
	cluster_aid * h, const size_t block, const ok_float maxdist)
{
	upsamplingvec * a2c_new = &h->a2c_tentative;
	matrix * A_blk = &h->A_reducible;
	cublasHandle_t * hdl = (cublasHandle_t *) h->hdl;

	size_t i;
	size_t row_stride_A = (A->order == CblasRowMajor) ? A->ld : 1;
	size_t col_stride_A = (A->order == CblasRowMajor) ? 1 : A->ld;
	size_t row_stride_C = (C->order == CblasRowMajor) ? C->ld : 1;
	size_t col_stride_C = (C->order == CblasRowMajor) ? 1 : A->ld;
	size_t ptr_stride = (A_blk->order == CblasRowMajor) ? A_blk->ld : 1;
	size_t idx_stride = (A_blk->order == CblasRowMajor) ? 1 : A_blk->ld;


	ok_status err = OPTKIT_SUCCESS;
	for (i = block; i < block + kBlockSize && i < A->size1 && !err; ++i) {
		size_t assignment_old, assignment_new;
		ok_float dmax;
		ok_float negOne = -kOne;
		int idxmax;
		cudaStream_t s;
		OK_CHECK_CUDA( err,
			cudaStreamCreate(&s) );
		OK_CHECK_CUDA( err,
			cudaMemcpyAsync(&assignment_old,
				a2c->indices + i * a2c->stride,
				sizeof(assignment_old), cudaMemcpyDeviceToHost,
				s) );
		OK_CHECK_CUDA( err,
			cudaMemcpyAsync(&assignment_new,
				a2c_new->indices + i * a2c_new->stride,
				sizeof(assignment_new), cudaMemcpyDeviceToHost,
				s) );
		if (assignment_old != assignment_new) {
			OK_CHECK_CUBLAS( err,
				cublasSetStream(*hdl, s) );
			OK_CHECK_CUBLAS( err,
				CUBLAS(copy)(*hdl, A->size2,
					A->data + i * row_stride_A,
					col_stride_A,
					A_blk->data + (i - block) * ptr_stride,
					idx_stride) );
			OK_CHECK_CUBLAS( err,
				CUBLAS(axpy)(*hdl, C->size2, &negOne,
					C->data + assignment_new * row_stride_C,
					col_stride_C,
					A_blk->data + (i - block) * ptr_stride,
					idx_stride) );
			OK_CHECK_CUBLAS( err,
				CUBLASI(amax)( *hdl, A->size2,
					A_blk->data + (i - block) * ptr_stride,
					idx_stride, &idxmax) );
			OK_CHECK_CUDA( err,
				cudaMemcpyAsync(&dmax, A_blk->data +
					(i - block) * ptr_stride +
					idxmax * idx_stride,
					sizeof(ok_float),
					cudaMemcpyDeviceToHost, s) );
			if (dmax <= maxdist) {
				OK_CHECK_CUDA( err,
					cudaMemcpyAsync(a2c->indices + i *
						a2c->stride, &assignment_new,
						sizeof(ok_float),
						cudaMemcpyHostToDevice, s) );
				++h->reassigned;
			}

		}
		OK_CHECK_CUDA( err,
			cudaStreamDestroy(s) );
	}
	cudaDeviceSynchronize();
	OK_MAX_ERR( err, OK_STATUS_CUDA );
	return err;
}

ok_status assign_clusters_l2_lInf_cap(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h, ok_float maxdist)
{
	ok_status err = OPTKIT_SUCCESS;
	uint block;

	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	OK_CHECK_UPSAMPLINGVEC(a2c);
	OK_CHECK_PTR(h);
	if (A->size1 != a2c->size1 || a2c->size2 > C->size1 ||
		A->size2 != C->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if (!h->A_reducible.data)
		err = matrix_calloc(&h->A_reducible, kBlockSize, A->size2,
			A->order);

	h->reassigned = 0;
	for (block = 0; block < A->size1 && !err; block += kBlockSize){
		OK_CHECK_ERR( err,
			assign_inner_loop(A, C, a2c, h, block, maxdist) );
	}

	OK_MAX_ERR( err, OK_SCAN_CUBLAS(
		cublasSetStream(*(cublasHandle_t *) h->hdl, NULL) ));

	return err;
}

#ifdef __cplusplus
}
#endif
