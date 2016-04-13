#ifndef OPTKIT_THRUST_H_
#define OPTKIT_THRUST_H_

#include "optkit_vector.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>


/*
 * strided iterator (from thrust:: examples)
 * =========================================
 */
template <typename Iterator>
class strided_range {
public:
	typedef typename thrust::iterator_difference<Iterator>::type diff_t;

	struct StrideF : public thrust::unary_function<diff_t, diff_t> {
		diff_t stride;
		StrideF(diff_t stride) : stride(stride) { }

		__host__ __device__ diff_t operator()(const diff_t& i) const {
			return stride * i;
		}
	};

	typedef typename thrust::counting_iterator<diff_t> CountingIt;
	typedef typename thrust::transform_iterator<StrideF, CountingIt>
		TransformIt;
	typedef typename thrust::permutation_iterator<Iterator, TransformIt>
		PermutationIt;

	  /* construct strided_range for the range [first,last). */
	strided_range(Iterator first, Iterator last, diff_t stride):
		first(first), last(last), stride(stride) { }

	PermutationIt begin() const {
	    return PermutationIt(first, TransformIt(CountingIt(0),
	    	StrideF(stride)));
	}

	PermutationIt end() const {
		return begin() + ((last - first) + (stride - 1)) / stride;
	}

 protected:
	Iterator first;
	Iterator last;
	diff_t stride;
};

typedef thrust::constant_iterator<ok_float> constant_iterator_t;
typedef strided_range<thrust::device_ptr<ok_float> > strided_range_t;

/*
 * thrust:: helper methods
 * =======================
 *
 * unary math ops
 * --------------
 */

/* x -> |x| */
struct AbsF : thrust::unary_function<ok_float, ok_float>
{
	__device__ inline ok_float operator()(ok_float x)
		{ return MATH(fabs)(x); }
};

/* x -> alpha / x */
struct ReciprF : thrust::unary_function<ok_float, ok_float>
{
	ok_float alpha;
	ReciprF() : alpha(1) {}
	ReciprF(ok_float alpha) : alpha(alpha) {}
	__device__ inline ok_float operator()(ok_float x)
		{ return alpha / x; }
};

/*
 * x -> alpha / x 	x != 0
 *		0 	otherwise
 */
struct SafeReciprF : thrust::unary_function<ok_float, ok_float>
{
	ok_float alpha;
	SafeReciprF() : alpha(1) {}
	SafeReciprF(ok_float alpha) : alpha(alpha) {}
	__device__ inline ok_float operator()(ok_float x)
		{ return ((ok_float) (x == kZero)) * alpha / x; }
};


/* x -> sqrt(x) */
struct SqrtF : thrust::unary_function<ok_float, ok_float>
{
        __device__ inline ok_float operator()(ok_float x)
                { return MATH(sqrt)(x); }
};

/* x -> x^p */
struct PowF : thrust::unary_function<ok_float, const ok_float>
{
	const ok_float p;
	PowF(const ok_float p) : p(p) {}
	__device__ inline ok_float operator()(ok_float x)
		{ return MATH(pow)(x, p); }
};

/* x -> exp(x) */
struct ExpF : thrust::unary_function<ok_float, ok_float>
{
	__device__ inline ok_float operator()(ok_float x)
		{ return MATH(exp)(x); }
};


/*
 *optkit.vector -> strided range
 * -----------------------------
 */
template<typename T>
inline strided_range<thrust::device_ptr<T> > __make_strided_range(
	vector_<T> * v)
{
	return strided_range_t(
		thrust::device_pointer_cast(v->data),
		thrust::device_pointer_cast(v->data + v->stride * v->size),
		v->stride);
}

template<typename T>
inline strided_range<thrust::device_ptr<T> > __make_const_strided_range(
	const vector_<T> * v)
{
	return strided_range_t(
		thrust::device_pointer_cast(v->data),
		thrust::device_pointer_cast(v->data + v->stride * v->size),
		v->stride);
}

/*
 * map unary/binary ops to strided range(s)
 * ----------------------------------------
 */

/* unary op mapped to (strided range) */
template <typename UnaryFunction>
inline void __transform_r(strided_range_t r, UnaryFunction f)
{
	thrust::transform(r.begin(), r.end(), r.begin(), f);
}

/* binary op mapped to (strided range, strided range) */
template <typename BinaryFunction>
inline void __transform_rr(strided_range_t r1, strided_range_t r2,
	BinaryFunction f)
{
	thrust::transform(r1.begin(), r1.end(), r2.begin(), r1.begin(), f);
}

/* binary op mapped to (strided range, constant) */
template <typename BinaryFunction>
inline void __transform_rc(strided_range_t r, ok_float x, BinaryFunction f)
{
	thrust::transform(r.begin(), r.end(), constant_iterator_t(x),
		r.begin(), f);
}

/* binary op mapped to (constant, strided range) */
template <typename BinaryFunction>
inline void __transform_cr(ok_float x, strided_range_t r, BinaryFunction f)
{
	thrust::transform(r.begin(), r.end(), constant_iterator_t(x),
		r.begin(), f);
}

/*
 * thrust:: elementwise optkit.vector operations
 * =============================================
 */

inline void __thrust_vector_scale(vector * v, ok_float x)
{
	strided_range_t r = __make_strided_range<ok_float>(v);
	__transform_rc(r, x, thrust::multiplies<ok_float>());
}

inline void __thrust_vector_add(vector * v1, const vector * v2)
{
	strided_range_t r1 = __make_strided_range<ok_float>(v1);
	strided_range_t r2 = __make_const_strided_range<ok_float>(v2);
	__transform_rr(r1, r2, thrust::plus<ok_float>());
}

inline void __thrust_vector_sub(vector * v1, const vector * v2)
{
	strided_range_t r1 = __make_strided_range<ok_float>(v1);
	strided_range_t r2 = __make_const_strided_range<ok_float>(v2);
	__transform_rr(r1, r2, thrust::minus<ok_float>());
}

inline void __thrust_vector_mul(vector * v1, const vector * v2)
{
	strided_range_t r1 = __make_strided_range<ok_float>(v1);
	strided_range_t r2 = __make_const_strided_range<ok_float>(v2);
	__transform_rr(r1, r2, thrust::multiplies<ok_float>());
}

inline void __thrust_vector_div(vector * v1, const vector * v2)
{
	strided_range_t r1 = __make_strided_range<ok_float>(v1);
	strided_range_t r2 = __make_const_strided_range<ok_float>(v2);
	__transform_rr(r1, r2, thrust::divides<ok_float>());
}

inline void __thrust_vector_add_constant(vector * v, const ok_float x)
{
	strided_range_t r = __make_strided_range<ok_float>(v);
	__transform_rc(r, x, thrust::plus<ok_float>());
}

inline void __thrust_vector_abs(vector * v)
{
	strided_range_t r = __make_strided_range<ok_float>(v);
	__transform_r(r, AbsF());
}

inline void __thrust_vector_recip(vector * v)
{
	strided_range_t r = __make_strided_range<ok_float>(v);
	__transform_r(r, ReciprF());
}

inline void __thrust_vector_safe_recip(vector * v)
{
	strided_range_t r = __make_strided_range(v);
	__transform_r(r, SafeReciprF());
}

inline void __thrust_vector_sqrt(vector * v)
{
	strided_range_t r = __make_strided_range<ok_float>(v);
	__transform_r(r, SqrtF());
}

inline void __thrust_vector_pow(vector * v, const ok_float p)
{
	strided_range_t r = __make_strided_range<ok_float>(v);
	__transform_r(r, PowF(p));
}

inline void __thrust_vector_exp(vector * v)
{
	strided_range_t r = __make_strided_range<ok_float>(v);
	__transform_r(r, ExpF(p));
}

template<typename T>
inline size_t __thrust_vector_indmin(vector_<T> * v)
{
	strided_range<thrust::device_ptr<T> > r = __make_strided_range<T>(v);
	thrust::device_ptr<T> res = thrust::min_element(r.begin(), r.end());
	return (size_t) ((res - v->data) / (v->stride * sizeof(T)));
}

template<typename T>
inline T __thrust_vector_min(vector_<T> * v)
{
	strided_range<thrust::device_ptr<T> > r = __make_strided_range<T>(v);
	thrust::device_ptr<T> res = thrust::min_element(r.begin(), r.end());
	return res[0];
}

template<typename T>
inline T __thrust_vector_min(vector_<T> * v)
{
	strided_range<thrust::device_ptr<T> > r = __make_strided_range<T>(v);
	thrust::device_ptr<ok_float> res = thrust::max_element(r.begin(),
		r.end());
	return res[0];
}

#endif /* OPTKIT_THRUST_H_ */
