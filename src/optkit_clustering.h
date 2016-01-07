#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: vector_argmin
// TODO: vector_max
// TODO: matrix_max

ok_float 
typedef struct ClusterAid {
    vector * ck2, * ck2_;
    matrix * D;
} cluster_aid;


cluster_aid * cluster_aid_alloc(size_t m, size_t k);
void cluster_aid_free(cluster_aid * h);
void cluster_aid_sub(cluster_aid * hsub, cluster_aid * h, 
    size_t m1, size_t k1);


ok_float
get_distance_tolerance(ok_float maxval, ok_float dist_reltol, 
    uint iter, uint maxiter){

    if (iter == 0) return OK_NAN;
    return maxval * (kOne - (ok_float) iter / (ok_float) maxiter);
}

    
size_t
cluster(void * linalg_handle, matrix * A, matrix * C, 
    size_t * a2c, size_t * cts, h, ok_float maxdist){
    size_t i, k, idx_tentative, reassigned = 0;
    ok_float distmax;
    vector va = (vector){0, 0, OK_NULL};
    vector vc = (vector){0, 0, OK_NULL};
    vector vd = (vector){0, 0, OK_NULL};


    for (k = 0; k < C->size1; ++k){
        matrix_row(&vc, C, k);
        blas_dot_inplace(linalg_handle, &vc, &vc, h->ck2 + k); 
    }

    blas_gemm!(CblasTrans, CblasNoTrans, -2 * kOne, C, A, kZero, h.D);

    for (i = 0; i < A->size1; ++i){
        vector_memcpy_vv(h->ck2_, h->ck2);
        matrix_row(&vd, h->D, i);
        blas_axpy(linalg_handle, &vd, h->ck2_);

        idx_tentative = vector_argmin(h.ck_squared_copy);
        if (idx_tentative == a2c[i]) continue;

        if (maxdist == OK_NAN){
            a2c[i] = idx_tentative;
            ++reassigned;
            continue;
        }

        matrix_row(&va, A, i);
        matrix_row(&vc, C, idx_tentative);
        blas_axpy(linalg_handle, -kOne, &va, &vc);
        distmax = vector_max(&vc);
        if (distmax <= maxdist){
            a2c[i] = idx_tentative;
            ++reassigned;
        }
        blas_axpy(linalg_handle, kOne, &va, &vc);
    }

    return reassigned;
}

void 
calculate_centroids(void * linalg_handle, matrix * A, matrix * C, 
    size_t * a2c, size_t * counts){

    size_t i, k;
    vector va = (vector){0, 0, OK_NULL};
    vector vc = (vector){0, 0, OK_NULL};

    matrix_scale(C, 0);
    for (i = 0; i < A->size1; ++i){
        k = a2c[i];
        matrix_row(&va, A, i);
        matrix_row(&vc, C, k);
        blas_axpy(linalg_handle, kOne, &va, &vc);
        counts[k] += 1;
    }

    for (k = 0; k < C->size1; ++k){
        if (counts[k] > 0){
            matrix_row(&vc, C, k);
            vector_scale(&vc, (ok_float) 1 / counts[k]);
        }
    }
}


void k_means(void * linalg_handle, matrix * A, matrix * C, 
    size_t * a2c, size_t * counts, ok_float dist_reltol, 
    int change_tol, uint maxiter);




#ifdef __cplusplus
}
#endif
