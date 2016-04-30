/* Adapted from from POGS (github.com/foges/pogs) */

#ifndef OPTKIT_PROXLIB_H_
#define OPTKIT_PROXLIB_H_

#include "optkit_defs.h"
#include "optkit_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OK_CHECK_FNVECTOR
#define OK_CHECK_FNVECTOR(f) \
	do { \
		if (!f || !f->objectives) \
			return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED ); \
	} while(0)
#endif

/* List of functions supported by the proximal operator library. */
enum OPTKIT_SCALAR_FUNCTION {
	FnZero, /* f(x) = 0 */
	FnAbs, /* f(x) = |x| */
	FnExp, /* f(x) = e^x */
	FnHuber, /* f(x) = huber(x) */
	FnIdentity, /* f(x) = x */
	FnIndBox01, /* f(x) = I(0 <= x <= 1) */
	FnIndEq0, /* f(x) = I(x = 0) */
	FnIndGe0, /* f(x) = I(x >= 0) */
	FnIndLe0, /* f(x) = I(x <= 0) */
	FnLogistic, /* f(x) = log(1 + e^x) */
	FnMaxNeg0, /* f(x) = max(0, -x) */
	FnMaxPos0, /* f(x) = max(0, x) */
	FnNegEntr, /* f(x) = x log(x) */
	FnNegLog, /* f(x) = -log(x) */
	FnRecipr, /* f(x) = 1/x */
	FnSquare /* f(x) = (1/2) x^2 */
};

#ifdef __cplusplus
}   /* extern "C" */
#endif


#ifdef __cplusplus
/* f(x) = c * h(ax-b) + dx + ex^2 */
template<typename T>
struct function_t_{
	enum OPTKIT_SCALAR_FUNCTION h;
	T a, b, c, d, e;
};

template<typename T>
struct function_vector_ {
	size_t size;
	function_t_<T> * objectives;
};
#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
typedef function_t_<ok_float> function_t;
typedef function_vector_<ok_float> function_vector;
#else
typedef struct function_t{
	enum OPTKIT_SCALAR_FUNCTION h;
	ok_float a, b, c, d, e;
} function_t;

typedef struct function_vector {
	size_t size;
	function_t * objectives;
} function_vector;
#endif


#ifdef __cplusplus
}   /* extern "C" */
#endif

#ifdef __cplusplus
__DEVICE__ inline double Abs(double x) { return fabs(x); }
__DEVICE__ inline double Acos(double x) { return acos(x); }
__DEVICE__ inline double Cos(double x) { return cos(x); }
__DEVICE__ inline double Exp(double x) { return exp(x); }
__DEVICE__ inline double Log(double x) { return log(x); }
__DEVICE__ inline double Max(double x, double y) { return fmax(x, y); }
__DEVICE__ inline double Min(double x, double y) { return fmin(x, y); }
__DEVICE__ inline double Pow(double x, double y) { return pow(x, y); }
__DEVICE__ inline double Sqrt(double x) { return sqrt(x); }

__DEVICE__ inline float Abs(float x) { return fabsf(x); }
__DEVICE__ inline float Acos(float x) { return acosf(x); }
__DEVICE__ inline float Cos(float x) { return cosf(x); }
__DEVICE__ inline float Exp(float x) { return expf(x); }
__DEVICE__ inline float Log(float x) { return logf(x); }
__DEVICE__ inline float Max(float x, float y) { return fmaxf(x, y); }
__DEVICE__ inline float Min(float x, float y) { return fminf(x, y); }
__DEVICE__ inline float Pow(float x, float y) { return powf(x, y); }
__DEVICE__ inline float Sqrt(float x) { return sqrtf(x); }

template<typename T>
__DEVICE__ inline T MaxPos(T x) { return Max(static_cast<T>(0), x); }

template<typename T>
__DEVICE__ inline T MaxNeg(ok_float x) { return Max(static_cast<T>(0), -x); }

template<typename T>
__DEVICE__ inline T Sign(T x) { return x >= 0 ? 1 : -1; }

/*
 * Evaluate the principal branch of the Lambert W function.
 *  ref: http: *keithbriggs.info/software/LambertW.c
 *
 * Specifically, evaluate LambertW(Exp(x)) since that is the only use case in
 * the sequal.
 */
template <typename T>
__DEVICE__ inline T LambertWExp(T x) {
	T w, log_x, p, e, t;
	if (x > static_cast<T>(100)) {

	/* Approximation for x in [100, 700]. */
	log_x = Log(x);
	return static_cast<T>(-0.36962844) + x
		- static_cast<T>(0.97284858) * log_x
		+ static_cast<T>(1.3437973) / log_x;
	} else if (x < static_cast<T>(0)) {
		p = Sqrt(static_cast<T>(2) *
			(Exp(x + static_cast<T>(1)) + static_cast<T>(1)));
		w = static_cast<T>(-1) + p * (static_cast<T>(1) +
			p * (static_cast<T>(-1.0 / 3.0) +
			p * static_cast<T>(11.0 / 72.0)));
	} else {
		w = x;
	}

	if (x > static_cast<T>(1.098612288668110))
		w -= Log(w);

	for (unsigned int i = 0u; i < 10u; i++) {
		e = Exp(w);
		t = w * e - Exp(x);
		p = w + static_cast<T>(1.);
		t /= e * p - static_cast<T>(0.5) *
			(p + static_cast<T>(1.0)) * t / p;
		w -= t;
	}
	return w;
}

/*
 * Find the root of a cubic x^3 + px^2 + qx + r = 0 with a single positive root.
 * ref: http: *math.stackexchange.com/questions/60376
 */
template<typename T>
__DEVICE__ inline T CubicSolve(T p, T q, T r)
{
	T s = p / 3, s2 = s * s, s3 = s2 * s;
	T a = -s2 + q / 3;
	T b = s3 - s * q / 2 + r / 2;
	T a3 = a * a * a;
	T b2 = b * b;

	if (a3 + b2 >= static_cast<T>(0)) {
		T A = Pow(Sqrt(a3 + b2) - b, static_cast<T>(1) / 3);
		return -s - a / A + A;
	} else {
		T A = Sqrt(-a3);
		T B = Acos(-b / A);
		T C = Pow(A, static_cast<T>(1) / 3);
		return -s + (C - a / C) * Cos(B / 3);
	}
}

/*
 * Proximal operator definitions.
 *
 * Each of the following functions corresponds to one of the Function enums.
 * All functions accept one argument x and five parameters (a, b, c, d and rho)
 * and returns the evaluation of
 *
 *   x -> Prox{c * f(a * x - b) + d * x + e * x ^ 2},
 *
 * where Prox{.} is the proximal operator with penalty parameter rho.
 */

template<typename T>
__DEVICE__ inline ok_float ProxAbs(ok_float v, ok_float rho)
{
	return MaxPos<T>(v - 1 / rho) - MaxNeg<T>(v + 1 / rho);
}

template<typename T>
__DEVICE__ inline ok_float ProxExp(ok_float v, ok_float rho)
{
	return v - LambertWExp<T>((v) / rho);
}

template<typename T>
__DEVICE__ inline ok_float ProxHuber(ok_float v, ok_float rho)
{
	return Abs(v) < 1 + 1 / rho ?
		v * rho / (1 + rho) : v - Sign<T>(v) / rho;
}

template<typename T>
__DEVICE__ inline ok_float ProxIdentity(ok_float v, ok_float rho)
{
	return v - 1 / rho;
}

template<typename T>
__DEVICE__ inline ok_float ProxIndBox01(ok_float v, ok_float rho)
{
	return v <= 0 ? 0 :
		v >= 1 ? 1 : v;
}

template<typename T>
__DEVICE__ inline ok_float ProxIndEq0(ok_float v, ok_float rho)
{
	return 0;
}

template<typename T>
__DEVICE__ inline ok_float ProxIndGe0(ok_float v, ok_float rho)
{
	return v <= 0 ? 0 : v;
}

template<typename T>
__DEVICE__ inline ok_float ProxIndLe0(ok_float v, ok_float rho)
{
	return v >= 0 ? 0 : v;
}

template<typename T>
__DEVICE__ inline ok_float ProxLogistic(ok_float v, ok_float rho)
{
	T x, l, inv_ex, f, g, g_rho, u;
	size_t i;

	/* Initial guess based on piecewise approximation. */
	if (v < static_cast<T>(-2.5))
		x = v;
	else if (v > (ok_float)(2.5) + static_cast<T>(1) / rho)
		x = v - static_cast<T>(1) / rho;
	else
		x = (rho * v - (ok_float)(0.5)) / ((ok_float)(0.2) + rho);

	/* Newton iteration. */
	l = v - static_cast<T>(1) / rho, u = v;
	for (i = 0; i < 5; ++i) {
		inv_ex = static_cast<T>(1) / (static_cast<T>(1) + Exp(-x));
		f = inv_ex + rho * (x - v);
		g = inv_ex * (static_cast<T>(1) - inv_ex) + rho;
		if (f < static_cast<T>(0))
			l = x;
		else
			u = x;

		x = x - f / g;
		x = Min(x, u);
		x = Max(x, l);
	}

	/* Guarded method if not converged. */
	for (i = 0; u - l > MACHINETOL && i < 100; ++i) {
		g_rho = static_cast<T>(1) /
			(rho * (static_cast<T>(1) + Exp(-x))) + (x - v);
		if (g_rho > static_cast<T>(0)) {
			l = Max(l, x - g_rho);
			u = x;
		} else {
			u = Min(u, x - g_rho);
			l = x;
		}
		x = (u + l) / static_cast<T>(2);
	}
	return x;
}

template<typename T>
__DEVICE__ inline T ProxMaxNeg0(T v, T rho)
{
	return v < -1. / rho ? v + 1. / rho : v > 0 ? v : 0;
}

template<typename T>
__DEVICE__ inline T ProxMaxPos0(T v, T rho)
{
	return v > 1. / rho ? v - 1. / rho : v < 0 ? v : 0;
}

template<typename T>
__DEVICE__ inline T ProxNegEntr(T v, T rho)
{
	return LambertWExp((rho * v - 1) * rho) / rho;
}

template<typename T>
__DEVICE__ inline T ProxNegLog(T v, T rho)
{
	return (v + Sqrt(v * v + static_cast<T>(4) / rho)) / 2;
}

template<typename T>
__DEVICE__ inline T ProxRecipr(T v, T rho)
{
	v = MaxPos<T>(v);
	return CubicSolve<T>(-v,  static_cast<T>(0), static_cast<T>(-1) / rho);
}

template<typename T>
__DEVICE__ inline T ProxSquare(T v, T rho) { return rho * v / (1 + rho); }

template<typename T>
__DEVICE__ inline T ProxZero(T v, T rho) { return v; }

/* Evaluates the proximal operator of f. */
template<typename T>
__DEVICE__ inline ok_float ProxEval(const function_t_<T> * f_obj, T v, T rho)
{
	const T a = f_obj->a;
	const T b = f_obj->b;
	const T c = f_obj->c;
	const T d = f_obj->d;
	const T e = f_obj->e;

	v = a * (v * rho - d) / (e + rho) - b;
	rho = (e + rho) / (c * a * a);

	switch ( f_obj->h ) {
	case FnAbs :
		v = ProxAbs<T>(v, rho);
		break;
	case FnExp :
		v = ProxExp<T>(v, rho);
		break;
	case FnHuber :
		v = ProxHuber<T>(v, rho);
		break;
	case FnIdentity :
		v = ProxIdentity<T>(v, rho);
		break;
	case FnIndBox01 :
		v = ProxIndBox01<T>(v, rho);
		break;
	case FnIndEq0 :
		v = ProxIndEq0<T>(v, rho);
		break;
	case FnIndGe0 :
		v = ProxIndGe0<T>(v, rho);
		break;
	case FnIndLe0 :
		v = ProxIndLe0<T>(v, rho);
		break;
	case FnLogistic :
		v = ProxLogistic<T>(v, rho);
		break;
	case FnMaxNeg0 :
		v = ProxMaxNeg0<T>(v, rho);
		break;
	case FnMaxPos0 :
		v = ProxMaxPos0<T>(v, rho);
		break;
	case FnNegEntr :
		v = ProxNegEntr<T>(v, rho);
		break;
	case FnNegLog :
		v = ProxNegLog<T>(v, rho);
		break;
	case FnRecipr :
		v = ProxRecipr<T>(v, rho);
		break;
	case FnSquare :
		v = ProxSquare<T>(v, rho);
		break;
	default :
		v = ProxZero<T>(v, rho);
		break;
	}
	return (v + b) / a;
}

/*
 * Function definitions.
 *
 * Each of the following functions corresponds to one of the Function enums.
 * All functions accept one argument x and four parameters (a, b, c, and d)
 * and returns the evaluation of
 *
 *   x -> c * f(a * x - b) + d * x.
*/
template<typename T>
__DEVICE__ inline T FuncAbs(T x) { return Abs(x); }

template<typename T>
__DEVICE__ inline T FuncExp(T x) { return Exp(x); }

template<typename T>
__DEVICE__ inline T FuncHuber(T x)
{
	T xabs = Abs(x);
	T xabs2 = xabs * xabs;
	return xabs < 1 ? xabs2 / 2 : xabs - 0.5;
}

template<typename T>
__DEVICE__ inline T FuncIdentity(T x) { return x; }

template<typename T>
__DEVICE__ inline T FuncIndBox01(T x) { return 0; }

template<typename T>
__DEVICE__ inline T FuncIndEq0(T x) { return 0; }

template<typename T>
__DEVICE__ inline T FuncIndGe0(T x) { return 0; }

template<typename T>
__DEVICE__ inline T FuncIndLe0(T x) { return 0; }

template<typename T>
__DEVICE__ inline T FuncLogistic(T x) { return Log(1 + Exp(x)); }

template<typename T>
__DEVICE__ inline T FuncMaxNeg0(T x) { return MaxNeg<T>(x); }

template<typename T>
__DEVICE__ inline T FuncMaxPos0(T x) { return MaxPos<T>(x); }

template<typename T>
__DEVICE__ inline T FuncNegEntr(T x)
{
	return x <= 0 ? static_cast<T>(0) : x * Log(x);
}

template<typename T>
__DEVICE__ inline T FuncNegLog(T x)
{
	return x > 0 ? -Log(x) : 0;
}

template<typename T>
__DEVICE__ inline T FuncRecipr(T x)
{
	return x > 0 ? 1 / x : 0;
}

template<typename T>
__DEVICE__ inline T FuncSquare(T x)
{
	return x * x / 2;
}

template<typename T>
__DEVICE__ inline T FuncZero(T x)
{
	return static_cast<T>(0);
}

/* Evaluates the function f. */
template<typename T>
__DEVICE__ inline T FuncEval(const function_t_<T> * f_obj, T x)
{
	T dx = f_obj->d * x;
	T ex = f_obj->e * x * x / 2;
	x = f_obj->a * x - f_obj->b;

	switch ( f_obj->h ) {
	case FnAbs:
		x = FuncAbs<T>(x);
		break;
	case FnExp:
		x = FuncExp<T>(x);
		break;
	case FnHuber:
		x = FuncHuber<T>(x);
		break;
	case FnIdentity:
		x = FuncIdentity<T>(x);
		break;
	case FnIndBox01:
		x = FuncIndBox01<T>(x);
		break;
	case FnIndEq0:
		x = FuncIndEq0<T>(x);
		break;
	case FnIndGe0:
		x = FuncIndGe0<T>(x);
		break;
	case FnIndLe0:
		x = FuncIndLe0<T>(x);
		break;
	case FnLogistic:
		x = FuncLogistic<T>(x);
		break;
	case FnMaxNeg0:
		x = FuncMaxNeg0<T>(x);
		break;
	case FnMaxPos0:
		x = FuncMaxPos0<T>(x);
		break;
	case FnNegEntr:
		x = FuncNegEntr<T>(x);
		break;
	case FnNegLog:
		x = FuncNegLog<T>(x);
		break;
	case FnRecipr:
		x = FuncRecipr<T>(x);
		break;
	case FnSquare:
		x = FuncSquare<T>(x);
		break;
	default:
		x = FuncZero<T>(x);
		break;
	}

	return f_obj->c * x + dx + ex;
}

template<typename T>
ok_status function_vector_alloc(function_vector_<T> * f, size_t n);
template<typename T>
ok_status function_vector_calloc_(function_vector_<T> * f, size_t n);
template<typename T>
ok_status function_vector_free_(function_vector_<T> * f);
template<typename T>
ok_status function_vector_view_array_(function_vector_<T> * f,
	function_t_<T> * h, size_t n);
template<typename T>
ok_status function_vector_memcpy_va_(function_vector_<T> * f,
	function_t_<T> * h);
template<typename T>
ok_status function_vector_memcpy_av_(function_t_<T> * h,
	function_vector_<T> * f);
template<typename T>
ok_status function_vector_mul_(function_vector_<T> * f, const vector_<T> * v);
template<typename T>
ok_status function_vector_div_(function_vector_<T> * f, const vector_<T> * v);
template<typename T>
ok_status function_vector_print_(function_vector_<T> *f);
template<typename T>
ok_status prox_eval_vector_(const function_vector_<T> * f, T rho,
	const vector_<T> * x_in, vector_<T> * x_out);
template<typename T>
ok_status function_eval_vector_(const function_vector_<T> * f,
	const vector_<T> * x, T * fn_val);
#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

ok_status function_vector_alloc(function_vector * f, size_t n);
ok_status function_vector_calloc(function_vector * f, size_t n);
ok_status function_vector_free(function_vector * f);
ok_status function_vector_view_array(function_vector * f, function_t * h, size_t n);
ok_status function_vector_memcpy_va(function_vector * f, function_t * h);
ok_status function_vector_memcpy_av(function_t * h, function_vector * f);
ok_status function_vector_mul(function_vector * f, const vector * v);
ok_status function_vector_div(function_vector * f, const vector * v);
ok_status function_vector_print(function_vector *f);
ok_status prox_eval_vector(const function_vector * f, ok_float rho,
	const vector * x_in, vector * x_out);
ok_status function_eval_vector(const function_vector * f, const vector * x,
	ok_float * fn_val);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_PROXLIB_H_ */
