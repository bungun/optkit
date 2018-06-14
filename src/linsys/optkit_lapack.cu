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
		cusolverDnDestroy(*(cusolverDnHandle_t *) lapack_handle) );
	ok_free(lapack_handle);
	return err;
}

ok_status lapack_LU_decomp_flagged(void *hdl, matrix *A, int_vector *pivot,
	int silence_cusolver_err)
{
	ok_status err = OPTKIT_SUCCESS;
	cusolverStatus_t cusolver_err;
	int dim_work, host_info;
	vector workspace = (vector){OK_NULL};
	int_vector info = (int_vector){OK_NULL};
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(pivot);
	OK_CHECK_PTR(hdl);

	if (A->size1 != A->size2 || pivot->size != A->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	cusolver_err = CUSOLVER(getrf_bufferSize)(*(cusolverDnHandle_t *) hdl,
		(int) A->size1, (int) A->size2, A->data, (int) A->ld, &dim_work);

	if (cusolver_err)
		return OK_SCAN_CUSOLVER(cusolver_err);

	OK_CHECK_ERR(err, vector_calloc(&workspace, (size_t) dim_work));
	OK_CHECK_ERR(err, int_vector_calloc(&info, 1));

	/* http://docs.nvidia.com/cuda/cusolver/index.html#ixzz53Sp2qt3h */
	if (!err)
		cusolver_err = CUSOLVER(getrf)(*(cusolverDnHandle_t *) hdl,
			(int) A->size1, (int) A->size2, A->data, (int) A->ld,
			workspace.data, pivot->data, info.data);

	if (cusolver_err && silence_cusolver_err)
		err = err > OPTKIT_ERROR_CUSOLVER ? err : OPTKIT_ERROR_CUSOLVER;
	else
		OK_MAX_ERR(err, OK_SCAN_CUSOLVER(cusolver_err));

	OK_MAX_ERR(err, int_vector_memcpy_av(&host_info, &info, 1));
	if (host_info != 0)
		err = err > OPTKIT_ERROR_CUSOLVER ? err : OPTKIT_ERROR_CUSOLVER;
		if (!silence_cusolver_err)
			OK_SCAN_ERR(OPTKIT_ERROR_CUSOLVER);
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
	int_vector dev_info = (int_vector){OK_NULL};
	int host_info;
	OK_CHECK_MATRIX(LU);
	OK_CHECK_MATRIX(X);
	OK_CHECK_PTR(hdl);

	if (LU->size1 != LU->size2 || X->size1 != LU->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	if (X->order != CblasColMajor)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );

	trans = (LU->order == CblasColMajor) ? CUBLAS_OP_N : CUBLAS_OP_T;

	OK_CHECK_ERR(err, int_vector_calloc(&dev_info, 1));
	OK_CHECK_CUSOLVER(err, CUSOLVER(getrs)(*(cusolverDnHandle_t *) hdl,
		trans, (int) LU->size1, (int) X->size2, LU->data, (int) LU->ld,
		pivot->data, X->data, (int) X->ld, dev_info.data));

	OK_CHECK_ERR(err, int_vector_memcpy_av(&host_info, &dev_info, 1));
	if (host_info != 0)
		OK_MAX_ERR(err, OPTKIT_ERROR_CUSOLVER);

	if (host_info < 0)
		printf("%s%i\n",
			"CUSOLVER generic triangular solve fail. Error at ith parameter, i=",
			-host_info);

	OK_MAX_ERR(err, int_vector_free(&dev_info));
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

ok_status lapack_cholesky_decomp_flagged(void *hdl, matrix *A,
	int silence_cusolver_err)
{
	ok_status err = OPTKIT_SUCCESS;
	cusolverStatus_t cusolver_err;
	cublasFillMode_t uplo;
	int dim_work, host_info;
	vector workspace = (vector){OK_NULL};
	int_vector info = (int_vector){OK_NULL};

	OK_CHECK_MATRIX(A);
	if (A->size1 != A->size2)
		return OK_SCAN_ERR(OPTKIT_ERROR_DIMENSION_MISMATCH);

	uplo = (A->order == CblasColMajor) ? CUBLAS_FILL_MODE_LOWER :
					     CUBLAS_FILL_MODE_UPPER;

	cusolver_err = CUSOLVER(potrf_bufferSize)(*(cusolverDnHandle_t *) hdl,
		uplo, (int) A->size1, A->data, (int) A->ld, &dim_work);
	if (cusolver_err)
		return OK_SCAN_CUSOLVER(cusolver_err);


	OK_CHECK_ERR(err, vector_calloc(&workspace, (size_t) dim_work));
	OK_CHECK_ERR(err, int_vector_calloc(&info, 1));
	if (!err)
		cusolver_err = CUSOLVER(potrf)(*(cusolverDnHandle_t *) hdl, uplo,
			(int) A->size1, A->data, (int) A->ld, workspace.data, dim_work,
			info.data);

	if (cusolver_err && silence_cusolver_err)
		err = err > OPTKIT_ERROR_CUSOLVER ? err : OPTKIT_ERROR_CUSOLVER;
	else
		OK_MAX_ERR(err, OK_SCAN_CUSOLVER(cusolver_err));

	OK_MAX_ERR(err, int_vector_memcpy_av(&host_info, &info, 1));
	if (host_info != 0)
		err = err > OPTKIT_ERROR_CUSOLVER ? err : OPTKIT_ERROR_CUSOLVER;
		if (!silence_cusolver_err)
			OK_SCAN_ERR(OPTKIT_ERROR_CUSOLVER);

	if (host_info && !silence_cusolver_err)
		printf("%s%i\n", "CUDA cholesky factorization failed: L_ii = 0 at i=",
			host_info);

	OK_MAX_ERR(err, vector_free(&workspace));
	OK_MAX_ERR(err, int_vector_free(&info));
	return err;
}

ok_status lapack_cholesky_svx(void *hdl, const matrix *L, vector *x)
{
	ok_status err = OPTKIT_SUCCESS;
	cublasFillMode_t uplo;
	int_vector dev_info = (int_vector){OK_NULL};
	int n, host_info;

	OK_CHECK_MATRIX(L);
	OK_CHECK_VECTOR(x);
	OK_CHECK_PTR(hdl);
	if (L->size1 != L->size2 || x->size != L->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	n = (int) L->size1;
	uplo = (L->order == CblasColMajor) ? CUBLAS_FILL_MODE_LOWER :
					     CUBLAS_FILL_MODE_UPPER;

	OK_CHECK_ERR(err, int_vector_calloc(&dev_info, 1));
	OK_CHECK_CUSOLVER(err, CUSOLVER(potrs)(*(cusolverDnHandle_t *) hdl,
		uplo, n, 1, L->data, n, x->data, n, dev_info.data));

	OK_CHECK_ERR(err, int_vector_memcpy_av(&host_info, &dev_info, 1));
	if (host_info != 0)
		OK_MAX_ERR(err, OPTKIT_ERROR_CUSOLVER);
	if (host_info < 0)
		printf("%s%i\n",
			"CUSOLVER cholesky solve fail. Error at ith parameter, i=",
			host_info);

	OK_MAX_ERR(err, int_vector_free(&dev_info));
	return err;
}

#ifdef __cplusplus
}
#endif

