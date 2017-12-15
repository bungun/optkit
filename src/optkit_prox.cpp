#include "optkit_prox.hpp"

template<typename T>
ok_status function_vector_alloc_(function_vector_<T> *f, size_t n)
{
	OK_CHECK_PTR(f);
	if (f->objectives)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	f->size = n;
	f->objectives = (function_t_<T> *) malloc(n *sizeof(*f->objectives));
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status function_vector_calloc_(function_vector_<T> *f, size_t n)
{
	size_t i;
	OK_RETURNIF_ERR( function_vector_alloc(f, n) );

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < n; ++i) {
		f->objectives[i].h = FnZero;
		f->objectives[i].a = kOne;
		f->objectives[i].b = kZero;
		f->objectives[i].c = kOne;
		f->objectives[i].d = kZero;
		f->objectives[i].e = kZero;
		f->objectives[i].s = kOne;
	}
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status function_vector_free_(function_vector_<T> *f)
{
	OK_CHECK_FNVECTOR(f);
	f->size = 0;
	ok_free(f->objectives);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status function_vector_view_array_(function_vector_<T> *f,
	function_t_<T> *h, size_t n)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_PTR(h);
	f->size = n;
	f->objectives = h;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status function_vector_memcpy_va_(function_vector_<T> *f,
	function_t_<T> *h)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_PTR(h);
	memcpy(f->objectives, h, f->size * sizeof(*h));
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status function_vector_memcpy_av_(function_t_<T> *h,
	function_vector_<T> *f)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_PTR(h);
	memcpy(h, f->objectives, f->size * sizeof(*h));
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status function_vector_mul_(function_vector_<T> *f, const vector_<T> *v)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_VECTOR(v);

	vector el;

	if (f->size != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	el.size = f->size;
	el.stride = sizeof(function_t_<T>) / sizeof(T);

	el.data = &(f->objectives->a);
	OK_RETURNIF_ERR( vector_mul(&el, v) );

	el.data = &(f->objectives->d);
	OK_RETURNIF_ERR( vector_mul(&el, v) );

	el.data = &(f->objectives->e);
	return OK_SCAN_ERR( vector_mul(&el, v) );
}

template<typename T>
ok_status function_vector_div_(function_vector_<T> *f, const vector_<T> *v)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_VECTOR(v);
	vector el;

	el.size = f->size;
	el.stride = sizeof(function_t_<T>) / sizeof(T);

	el.data = &(f->objectives->a);
	OK_RETURNIF_ERR( vector_div(&el, v) );

	el.data = &(f->objectives->d);
	OK_RETURNIF_ERR( vector_div(&el, v) );

	el.data = &(f->objectives->e);
	return OK_SCAN_ERR( vector_div(&el, v) );
}

template<typename T>
ok_status function_vector_print_(function_vector_<T> *f)
{
	OK_CHECK_FNVECTOR(f);
	size_t i;
	const char *fmt =
		"h: %i, a:%1.2e, b:%1.2e, c:%1.2e, d:%1.2e, e:%1.2e, s:%1.2e\n";
	for (i = 0; i < f->size; ++i)
		printf(fmt, (int) f->objectives[i].h, f->objectives[i].a,
			f->objectives[i].b, f->objectives[i].c,
			f->objectives[i].d, f->objectives[i].e,
			f->objectives[i].s);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status prox_eval_vector_(const function_vector_<T> *f, T rho,
			  const vector_<T> *x_in, vector_<T> *x_out)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_VECTOR(x_in);
	OK_CHECK_VECTOR(x_out);
	if (f->size != x_in->size || f->size != x_out->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	if (rho <= 0)
		return OK_SCAN_ERR( OPTKIT_ERROR_DOMAIN );

	uint i;
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < f->size; ++i)
		x_out->data[i * x_out->stride] = ProxEval<T>(&f->objectives[i],
			x_in->data[i * x_in->stride], rho);

	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status function_eval_vector_(const function_vector_<T> *f,
	const vector_<T> *x, T * fn_val)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_VECTOR(x);
	OK_CHECK_PTR(fn_val);
	if (f->size != x->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	T sum = 0;
	uint i;
	#ifdef _OPENMP
	#pragma omp parallel for reduction(+:sum)
	#endif
	for (i = 0; i < f->size; ++i)
		sum += FuncEval<T>(&f->objectives[i], x->data[i * x->stride]);
	*fn_val = sum;

	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

ok_status function_vector_alloc(function_vector *f, size_t n)
	{ return function_vector_alloc_<ok_float>(f, n); }

ok_status function_vector_calloc(function_vector *f, size_t n)
	{ return function_vector_calloc_<ok_float>(f, n); }

ok_status function_vector_free(function_vector *f)
	{ return function_vector_free_<ok_float>(f); }

ok_status function_vector_view_array(function_vector *f, function_t *h,
	size_t n)
	{ return function_vector_view_array_<ok_float>(f, h, n); }

ok_status function_vector_memcpy_va(function_vector *f, function_t *h)
	{ return function_vector_memcpy_va_<ok_float>(f, h); }

ok_status function_vector_memcpy_av(function_t *h, function_vector *f)
	{ return function_vector_memcpy_av_<ok_float>(h, f); }

ok_status function_vector_mul(function_vector *f, const vector *v)
	{ return function_vector_mul_<ok_float>(f, v); }

ok_status function_vector_div(function_vector *f, const vector *v)
	{ return function_vector_div_<ok_float>(f, v); }

ok_status function_vector_print(function_vector *f)
	{ return function_vector_print_<ok_float>(f); }

ok_status prox_eval_vector(const function_vector *f, ok_float rho,
	const vector *x_in, vector *x_out)
	{ return prox_eval_vector_<ok_float>(f, rho, x_in, x_out); }
ok_status function_eval_vector(const function_vector *f, const vector *x,
	ok_float *fn_val)
	{ return function_eval_vector_<ok_float>(f, x, fn_val); }

#ifdef __cplusplus
}
#endif
