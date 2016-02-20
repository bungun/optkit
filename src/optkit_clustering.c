#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

cluster_aid * cluster_aid_alloc(size_t m, size_t k)
{
        cluster_aid * h;
        h = malloc(sizeof(*h));
        h->ck2 = (vector *) malloc(sizeof(vector));
        h->ck2_ = (vector *) malloc(sizeof(vector));
        h->D = (matrix *) malloc(sizeof(matrix));

        vector_calloc(h->ck2, k);
        vector_calloc(h->ck2_, k);
        matrix_calloc(h->D, m, k);

        return h;
}

void cluster_aid_free(cluster_aid * h)
{
        vector_free(h->ck2);
        vector_free(h->ck2_);
        matrix_free(h->D);
        ok_free(h->ck2);
        ok_free(h->ck2_);
        ok_free(h->D);
        ok_free(h);
}

void cluster_aid_sub(cluster_aid * hsub, cluster_aid * h, size_t m1, size_t k1)
{
        vector_subvector(hsub->ck2, h->ck2, 0, k1);
        vector_subvector(hsub->ck2_, h->ck2_, 0, k1);
        matrix_submatrix(hsub->D, h->D, 0, 0, m1, k1);
}

void k_means(void * linalg_handle, matrix * A, matrix * C, size_t * a2c,
        size_t * counts, ok_float dist_reltol, int change_tol, uint maxiter)
{
        size_t i, reassigned;
        ok_float tol, maxA = matrix_max(A);

        for (i = 0; i < maxiter; ++i) {
                tol = get_distance_tolerance(maxA, dist_reltol, i, maxiter);
                reassigned = cluster(linalg_handle, A, C, a2c, counts);
                calculate_centroids(linalg_handle, A, C, a2c, counts);
                if (reassigned < change_tol) {
                        printf("iteration: %u \t changed: $(h.change_sum) \
                                \t change tol: $(CHANGE_TOL)",
                                iter, reassigned, change_tol);
                            printf("quitting k-means after %u iterations\n",
                                iter);
                    break;
                } else {
                        printf("iteration: %u \t changed: %u \t change tol: %u",
                                iter, reassigned, change_tol);
                }
        }
}

#ifdef __cplusplus
}
#endif