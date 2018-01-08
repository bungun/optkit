#include "optkit_defs_gpu.h"
#include "optkit_lapack.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status lapack_make_handle(void **lapack_handle)
{
	ok_status err = OPTKIT_SUCCESS;
	cusolverDnHandle_t *hdl;
	hdl = (cusolverDnHandle_t *) malloc(sizeof(cusolverDnHandle_t));
	err = OK_SCAN_CUSOLVER( cusolverDnCreate(hdl) );
	if (err) {
		printf("%s\n", "CUSOLVER initialization failed");
		ok_free(hdl);
		*lapack_handle = OK_NULL;
		return err;
	} else {
		*lapack_handle = (void *) hdl;
		return OPTKIT_SUCCESS;
	}
}

ok_status lapack_destroy_handle(void *lapack_handle)
{
	OK_CHECK_PTR(lapack_handle);
	ok_status err = OK_SCAN_CUSOLVER(
		cusolverDnDestroy(*(cusolverDnHandle_t *) handle) );
	ok_free(lapack_handle);
	return err;
}

ok_status lapack_LU_decomp_flagged(void *hdl, matrix *A, int_vector *pivot,
	int silence_cusolver_err);
ok_status lapack_LU_decomp(void *hdl, matrix *A, int_vector *pivot);
ok_status lapack_LU_svx(void *hdl, matrix *LU, vector *x);

ok_status lapack_LU_decomp_flagged(void *hdl, matrix *A, int_vector *pivot,
	int silence_cusolver_err)
{
	ok_status err = OPTKIT_SUCCESS;
	cusolverStatus_t cusolver_err;
	int dim_work, host_info;
	vector workspace;
	int_vector info;
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(pivot);
	OK_CHECK_PTR(hdl);

	if (A->size1 != A->size2 || pivot->size != A->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );


	cusolver_err = CUSOLVER(getrf_bufferSize)(*(cusolverDnHandle_t *) handle,
		(int) A->size1, (int) A->size2, A->data, (int) A->ld, &dim_work);
	if (cusolver_err)
		return OK_SCAN_CUSOLVER(cusolver_err);

	OK_CHECK_ERR(err, vector_calloc(&workspace, (size_t) dim_work));
	OK_CHECK_ERR(err, int_vector_calloc(&info, 1));

	/* http://docs.nvidia.com/cuda/cusolver/index.html#ixzz53Sp2qt3h */
	cusolver_err = CUSOLVER(getrf)(*(cusolverDnHandle_t *) handle,
		(int) A->size1, (int) A->size2, A->data, (int) A->ld,
		workspace->data, pivot->data, info->data);

	if (cusolver_err && silence_cusolver_err)
		err = OPTKIT_ERROR_CUSOLVER;
	else
		err = OK_SCAN_CUSOLVER(cusolver_err);

	OK_MAX_ERR(err, int_vector_memcpy_av(&host_info, &info, 1));
	if (host_info && !silence_cusolver_err)
		printf("%s%i\n", "CUDA LU factorization failed: U_ii = 0 at i=",
			host_info);

	OK_MAX_ERR(err, vector_free(&workspace));
	OK_MAX_ERR(err, int_vector_free(&info));
	return err;
}

ok_status lapack_LU_decomp(void *hdl, matrix *A, int_vector *pivot)
{
	return OK_SCAN_ERR(lapack_LU_decomp_flagged(hdl, A, pivot, 0));
}

ok_status lapack_LU_svx(void *hdl, matrix *LU, matrix *X, int_vector *pivot)
{
	ok_status err = OPTKIT_SUCCESS;
	cublasOperation_t trans;
	int_vector dev_info;
	int host_info;
	OK_CHECK_MATRIX(LU);
	OK_CHECK_MATRIX(X);
	OK_CHECK_PTR(hdl);

	if (LU->size1 != LU->size2 || X->size1 != LU->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	if (X->ord != CblasColMajor)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );

	trans = (LU->ord == CblasColMajor) ? CUBLAS_OP_N : CUBLAS_OP_T;

	err = OK_SCAN_CUSOLVER(CUSOLVER(getrs)(*(cusolverDnHandle_t *) handle,
		trans, (int) LU->size1, (int) X->size2, LU->data, (int) LU->ld,
		pivot->data, X->data, (int) X->ld, info->data));

	OK_MAX_ERR(err, vector_memcpy_av(&host_info, &dev_info, 1));
	if (host_info < 1)
		printf("%s%i\n",
			"CUSOLVER generic triangular solve fail. Error at ith parameter, i=",
			-host_info);
	return err;
}

ok_status lapack_solve_LU_matrix_flagged(void *hdl, matrix *A, matrix *X,
	int_vector *pivot, int silence_lapack_err)
{
	ok_status err = OPTKIT_SUCCESS;

	if (silence_lapack_err)
		err = lapack_LU_decomp_flagged(hdl, A, pivot, 1);
	else
		err = OK_SCAN_ERR(lapack_LU_decomp_flagged(hdl, A, pivot, 0));

	OK_CHECK_ERR(err, lapack_LU_svx(hdl, A, X, pivot));
	return err;
}

#ifdef __cplusplus
}
#endif
