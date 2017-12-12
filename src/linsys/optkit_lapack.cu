#include "optkit_defs_gpu.h"
#include "optkit_lapack.h"

#ifdef __cplusplus
extern "C" {
#endif


// ok_status lapack_make_handle(void **lapack_handle)
// {
//     return OPTKIT_SUCCESS;
// }

// ok_status lapack_destroy_handle(void *lapack_handle)
// {
//     return OPTKIT_SUCCESS;
// }

// ok_status lapack_solve_LU(void * hdl, matrix *A, vector *x, int_vector *pivot)
// {
//     lapack_int *pivot = OK_NULL;
//     lapack_int err = 0;

//     OK_CHECK_MATRIX(A);
//     OK_CHECK_VECTOR(x);
//     if (A->size1 != A->size2 || x->size != A->size2)
//         return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

//     ok_alloc(pivot, sizeof(*pivot) * A->size1);
//     err = LAPACK(gesv)((int) A->order, (lapack_int) A->size1,
//         (lapack_int) 1, A->data, (int) A->ld, pivot, x->data, (int) x->stride);
//     ok_free(pivot);

//     if (err) {
//         printf("%s%i\n", "LAPACK ERROR: ", (int) err);
//         return OK_SCAN_ERR( OPTKIT_ERROR_LAPACK );
//     }
//     return OPTKIT_SUCCESS;
// }

// ok_status lapack_solve_LU_matrix(void * hdl, matrix *A, matrix *X, int_vector *pivot)
// {
//     lapack_int *pivot = OK_NULL;
//     lapack_int err = 0;

//     OK_CHECK_MATRIX(A);
//     OK_CHECK_MATRIX(X);

//     if (A->size1 != A->size2 || X->size1 != A->size2)
//         return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
//     if (A->order != X->order)
//         return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );

//     ok_alloc(pivot, sizeof(*pivot) * A->size1);
//     err = LAPACK(gesv)((int) A->order, (lapack_int) A->size1,
//         (lapack_int) X->size2, A->data, (int) A->ld, (lapack_int *) pivot,
//         X->data, (int) X->ld);
//     ok_free(pivot);

//     if (err) {
//         printf("%s%i\n", "LAPACK ERROR: ", (int) err);
//         return OK_SCAN_ERR( OPTKIT_ERROR_LAPACK );
//     }
//     return OPTKIT_SUCCESS;
// }

#ifdef __cplusplus
}
#endif
