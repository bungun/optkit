#include "optkit_upsampling_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status upsamplingvec_alloc(upsamplingvec * u, size_t size1, size_t size2)
{
	if (!u)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	else if (u->indices)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	memset(u, 0, sizeof(*u));
	u->size1 = size1;
	u->size2 = size2;
	OK_RETURNIF_ERR( indvector_calloc(&u->vec, size1) );
	u->indices = u->vec.data;
	u->stride = u->vec.stride;
	return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_free(upsamplingvec * u)
{
	OK_CHECK_UPSAMPLINGVEC(u);
	OK_RETURNIF_ERR( indvector_free(&u->vec) );
	memset(u, 0, sizeof(*u));
	return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_check_bounds(const upsamplingvec * u)
{
	size_t idx;
	OK_CHECK_UPSAMPLINGVEC(u);
	OK_RETURNIF_ERR( indvector_max(&u->vec, &idx) );
	if (idx >= u->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	else
		return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_update_size(upsamplingvec * u)
{
	OK_CHECK_UPSAMPLINGVEC(u);
	ok_status err = indvector_max(&u->vec, &u->size2);
	++u->size2;
	return err;
}

ok_status upsamplingvec_subvector(upsamplingvec * usub, upsamplingvec * u,
	size_t offset1, size_t length1, size_t size2)
{
	if (!u || !u->indices || !usub)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	OK_RETURNIF_ERR( indvector_subvector(&usub->vec, &u->vec, offset1,
		length1) );
	usub->indices = usub->vec.data;
	usub->size1 = length1;
	usub->size2 = size2;
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
