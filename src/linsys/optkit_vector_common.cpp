#include "optkit_vector.h"

template<typename T>
ok_status vector_subvector_(vector_<T> *v_out, vector_<T> *v_in, size_t offset,
        size_t n)
{
        if (!v_out || !v_in || !v_in->data)
                return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
        if (v_out->data)
                return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
        v_out->size=n;
        v_out->stride=v_in->stride;
        v_out->data=v_in->data + offset * v_in->stride;
        return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_view_array_(vector_<T> *v, T *base, size_t n)
{
        if (!v || !base)
                return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
        if (v->data)
                return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

        v->size=n;
        v->stride=1;
        v->data=base;
        return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif


/* VECTOR */
ok_status vector_alloc(vector *v, size_t n)
        { return vector_alloc_<ok_float>(v, n); }

ok_status vector_calloc(vector *v, size_t n)
        { return vector_calloc_<ok_float>(v, n); }

ok_status vector_free(vector *v)
        { return vector_free_<ok_float>(v); }

ok_status vector_set_all(vector *v, ok_float x)
        { return vector_set_all_<ok_float>(v, x); }

ok_status vector_subvector(vector *v_out, vector *v_in, size_t offset, size_t n)
        { return vector_subvector_<ok_float>(v_out, v_in, offset, n); }

ok_status vector_view_array(vector *v, ok_float *base, size_t n)
        { return vector_view_array_<ok_float>(v, base, n); }

ok_status vector_memcpy_vv(vector *v1, const vector *v2)
        { return vector_memcpy_vv_<ok_float>(v1, v2); }

ok_status vector_memcpy_va(vector *v, const ok_float *y, size_t stride_y)
        { return vector_memcpy_va_<ok_float>(v, y, stride_y); }

ok_status vector_memcpy_av(ok_float *x, const vector *v, size_t stride_x)
        { return vector_memcpy_av_<ok_float>(x, v, stride_x); }

ok_status vector_add(vector *v1, const vector *v2)
        { return vector_add_<ok_float>(v1, v2); }

ok_status vector_sub(vector *v1, const vector *v2)
        { return vector_sub_<ok_float>(v1, v2); }

ok_status vector_mul(vector *v1, const vector *v2)
        { return vector_mul_<ok_float>(v1, v2); }

ok_status vector_div(vector *v1, const vector *v2)
        { return vector_div_<ok_float>(v1, v2); }

ok_status vector_add_constant(vector *v, const ok_float x)
        { return vector_add_constant_<ok_float>(v, x); }

ok_status vector_indmin(const vector *v, size_t *idx)
        { return vector_indmin_<ok_float>(v, (ok_float) OK_FLOAT_MAX, idx); }

ok_status vector_min(const vector *v, ok_float *minval)
        { return vector_min_<ok_float>(v, (ok_float) OK_FLOAT_MAX, minval); }

ok_status vector_max(const vector *v, ok_float *maxval)
        { return vector_max_<ok_float>(v, (ok_float) -OK_FLOAT_MAX, maxval); }

/* INDVECTOR */
ok_status indvector_alloc(indvector *v, size_t n)
        { return vector_alloc_<size_t>(v, n); }

ok_status indvector_calloc(indvector *v, size_t n)
        { return vector_calloc_<size_t>(v, n); }

ok_status indvector_free(indvector *v)
        { return vector_free_<size_t>(v); }

ok_status indvector_set_all(indvector *v, size_t x)
        { return vector_set_all_<size_t>(v, x); }

ok_status indvector_subvector(indvector *v_out, indvector *v_in,
        size_t offset, size_t n)
        { return vector_subvector_<size_t>(v_out, v_in, offset, n); }

ok_status indvector_view_array(indvector *v, size_t *base, size_t n)
        { return vector_view_array_<size_t>(v, base, n); }

ok_status indvector_memcpy_vv(indvector *v1, const indvector *v2)
        { return vector_memcpy_vv_<size_t>(v1, v2); }

ok_status indvector_memcpy_va(indvector *v, const size_t *y, size_t stride_y)
        { return vector_memcpy_va_<size_t>(v, y, stride_y); }

ok_status indvector_memcpy_av(size_t *x, const indvector *v, size_t stride_x)
        { return vector_memcpy_av_<size_t>(x, v, stride_x); }

ok_status indvector_indmin(const indvector *v, size_t *idx)
        { return vector_indmin_<size_t>(v, (size_t) INT_MAX, idx); }

ok_status indvector_min(const indvector *v, size_t *minval)
        { return vector_min_<size_t>(v, (size_t) INT_MAX, minval); }

ok_status indvector_max(const indvector *v, size_t *maxval)
        { return vector_max_<size_t>(v, 0, maxval); }

/* INT_VECTOR */
ok_status int_vector_alloc(int_vector *v, size_t n)
        { return vector_alloc_<ok_int>(v, n); }

ok_status int_vector_calloc(int_vector *v, size_t n)
        { return vector_calloc_<ok_int>(v, n); }

ok_status int_vector_free(int_vector *v)
        { return vector_free_<ok_int>(v); }

ok_status int_vector_memcpy_vv(int_vector *v1, const int_vector *v2)
        { return vector_memcpy_vv_<ok_int>(v1, v2); }

ok_status int_vector_memcpy_va(int_vector *v, const ok_int *y, size_t stride_y)
        { return vector_memcpy_va_<ok_int>(v, y, stride_y); }

ok_status int_vector_memcpy_av(ok_int *x, const int_vector *v, size_t stride_x)
        { return vector_memcpy_av_<ok_int>(x, v, stride_x); }


#ifdef __cplusplus
}
#endif
