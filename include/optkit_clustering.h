#ifndef OPTKIT_CLUSTERING_H_
#define OPTKIT_CLUSTERING_H_

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cluster_aid {
	void * hdl, /* linalg handle */
	vector a, c, d, c_squared_full, c_squared, c_tentative,
        matrix D_full, D,
        size_t reassigned, idx_tentative,
        ok_float maxdist,
        int freeable
} cluster_aid;

typedef struct upsamplingvec {
	size_t * vals,
	size_t size, shift
};

void upsamplingvec_alloc(upsamplingvec * u, size_t size);
void upsamplingvec_calloc(upsamplingvec * u, size_t size);
void upsamplingvec_free(upsamplingvec * u);
void upsamplingvec_subvector(upsamplingvec * usub, upsamplingvec * u,
	size_t offset, size_t size, size_t shift);

void cluster_aid_init(matrix * A, matrix * C);
void cluster_aid_finish(cluster_aid * h);
void cluster(matrix * A, matrix * C, upsamplingvec * a2c, cluster_aid ** helper,
	ok_float maxdist);
void calculate_centroids(matrix * A, matrix * C, upsamplingvec * a2c,
	vector * counts, cluster_aid * h);

void k_means(matrix * A, matrix * C, upsamplingvec * a2c, vector * counts,
	cluster_aid * h, const ok_float dist_reltol, const int change_tol,
	const size_t maxiter, const uint verbose);

void blockwise_kmeans(matrix * A, matrix * C, upsamplingvec * a2c,
	vector * counts, const size_t n_blocks, const size_t * blocksA,
	const size_t * blocksC, const size_t maxiter, const ok_float dist_tol,
	const ok_float * change_tols, const uint verbose);



#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_CLUSTERING_H_ */
