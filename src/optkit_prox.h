/* Adapted to C from from POGS (github.com/foges/pogs) */


#ifndef OPTKIT_PROXLIB_H_GUARD
#define OPTKIT_PROXLIB_H_GUARD

#include "optkit_defs.h"


#ifdef __cplusplus
extern "C" {
#endif



/* List of functions supported by the proximal operator library. */
enum Function { 
                FnZero 		/* f(x) = 0 				*/
				FnAbs,       /* f(x) = |x| 				*/
                FnExp,       /* f(x) = e^x 				*/
                FnHuber,     /* f(x) = huber(x) 			*/
                FnIdentity,  /* f(x) = x 				*/
                FnIndBox01,  /* f(x) = I(0 <= x <= 1) 	*/
                FnIndEq0,    /* f(x) = I(x = 0) 			*/
                FnIndGe0,    /* f(x) = I(x >= 0) 		*/
                FnIndLe0,    /* f(x) = I(x <= 0) 		*/
                FnLogistic,  /* f(x) = log(1 + e^x) 		*/
                FnMaxNeg0,   /* f(x) = max(0, -x) 		*/
                FnMaxPos0,   /* f(x) = max(0, x)  		*/
                FnNegEntr,   /* f(x) = x log(x) 			*/
                FnNegLog,    /* f(x) = -log(x) 			*/
                FnRecipr,    /* f(x) = 1/x  				*/
                FnSquare,    /* f(x) = (1/2) x^2 		*/
                };    

typedef enum Function Function_t

/* f(x) = c * h(ax-b) + dx + ex^2 */
typedef struct FunctionObj{
	Function_t h;
	ok_float a, b, c, d, e;
} FunctionObj;

typedef struct FunctionVector{
	size_t size;
	FunctionObj * objectives;
} FunctionVector;


/* check/enforce convexity of function objects */
void checkvexity(FunctionObj * f);
/*void init_fobj(FunctionObj * f);
void init_fobj_a(FunctionObj * f, Function_t h, ok_float a);
void init_fobj_ab(FunctionObj * f, Function_t h, 
				  ok_float a, ok_float b);
void init_fobj_ac(FunctionObj * f, Function_t h, 
				  ok_float a, ok_float b, ok_float c);
void init_fobj_ad(FunctionObj * f, Function_t h, 
				  ok_float a, ok_float b,
				  ok_float c, ok_float d);
void init_fobj_ae(FunctionObj * f, Function_t h, 
				  ok_float a, ok_float b,
				  ok_float c, ok_float d,
				  ok_float e);*/
void function_vector_alloc(FunctionVector * f, size_t len);
void function_vector_calloc(FunctionVector * f, size_t len);
void function_vector_free(FunctionVector * f);

__DEVICE__ inline ok_float Abs(ok_float x) { return fabs(x); }
__DEVICE__ inline ok_float Acos(ok_float x) { return acos(x); }
__DEVICE__ inline ok_float Cos(ok_float x) { return cos(x); }
__DEVICE__ inline ok_float Exp(ok_float x) { return exp(x); }
__DEVICE__ inline ok_float Log(ok_float x) { return log(x); }
__DEVICE__ inline ok_float Max(ok_float x, ok_float y) { return fmax(x, y); }
__DEVICE__ inline ok_float Min(ok_float x, ok_float y) { return fmin(x, y); }
__DEVICE__ inline ok_float Pow(ok_float x, ok_float y) { return pow(x, y); }
__DEVICE__ inline ok_float Sqrt(ok_float x) { return sqrt(x); }


__DEVICE__ inline ok_float MaxPos(ok_float x) {
  return Max((ok_float) 0, x);
}

__DEVICE__ inline ok_float MaxNeg(ok_float x) {
  return Max((ok_float) 0, -x);
}

__DEVICE__ inline ok_float Sign(ok_float x) {
  return x >= 0 ? 1 : -1;
}

/* Evaluate the principal branch of the Lambert W function.
   ref: http://keithbriggs.info/software/LambertW.c */
__DEVICE__ inline ok_float LambertW(ok_float x) {
  const ok_float kEm1 = (ok_float)(0.3678794411714423215955237701614608);
  const ok_float kE = (ok_float)(2.7182818284590452353602874713526625);
  if (x == 0) {
    return 0;
  } else if (x < -kEm1 + 1e-4) {
    ok_float q = x + kEm1, r = Sqrt(q), q2 = q * q, q3 = q2 * q;
    return
     -(ok_float) 1.0
     +(ok_float) 2.331643981597124203363536062168 * r
     -(ok_float) 1.812187885639363490240191647568 * q
     +(ok_float) 1.936631114492359755363277457668 * r * q
     -(ok_float) 2.353551201881614516821543561516 * q2
     +(ok_float) 3.066858901050631912893148922704 * r * q2
     -(ok_float) 4.175335600258177138854984177460 * q3
     +(ok_float) 5.858023729874774148815053846119 * r * q3
     -(ok_float) 8.401032217523977370984161688514 * q3 * q;
  } else {
    ok_float w;
    if (x < 1) {
      ok_float p = Sqrt((ok_float)(2.0 * (kE * x + 1.0)));
      w = (ok_float)(-1.0 + p * (1.0 + p * (-1.0 / 3.0 + p * 11.0 / 72.0)));
    } else {
      w = Log(x);
    }
    if (x > 3)
      w -= Log(w);
    for (unsigned int i = 0; i < 10; i++) {
      ok_float e = Exp(w);
      ok_float t = w * e - x;
      ok_float p = w + (ok_float)(1);
      t /= (ok_float)(e * p - 0.5 * (p + 1.0) * t / p);
      w -= t;
    }
    return w;
  }
}

/* Find the root of a cubic x^3 + px^2 + qx + r = 0 with a single positive root.
   ref: http://math.stackexchange.com/questions/60376 */
__DEVICE__ inline ok_float CubicSolve(ok_float p, ok_float q, ok_float r) {
  ok_float s = p / 3, s2 = s * s, s3 = s2 * s;
  ok_float a = -s2 + q / 3;
  ok_float b = s3 - s * q / 2 + r / 2;
  ok_float a3 = a * a * a;
  ok_float b2 = b * b;
  if (a3 + b2 >= 0) {
    ok_float A = Pow(Sqrt(a3 + b2) - b, (ok_float)(1) / 3);
    return -s - a / A + A;
  } else {
    ok_float A = Sqrt(-a3);
    ok_float B = Acos(-b / A);
    ok_float C = Pow(A, (ok_float)(1) / 3);
    return -s + (C - a / C) * Cos(B / 3);
  }
}


/*
// Proximal operator definitions.
//
// Each of the following functions corresponds to one of the Function enums.
// All functions accept one argument x and five parameters (a, b, c, d and rho)
// and returns the evaluation of
//
//   x -> Prox{c * f(a * x - b) + d * x + e * x ^ 2},
//
// where Prox{.} is the proximal operator with penalty parameter rho.
*/

__DEVICE__ inline ok_float ProxAbs(ok_float v, ok_float rho) {
  return MaxPos(v - 1 / rho) - MaxNeg(v + 1 / rho);
}

__DEVICE__ inline ok_float ProxNegEntr(ok_float v, ok_float rho) {
  return LambertW(Exp(rho * v - 1) * rho) / rho;
}

__DEVICE__ inline ok_float ProxExp(ok_float v, ok_float rho) {
  return v - LambertW(Exp(v) / rho);
}

__DEVICE__ inline ok_float ProxHuber(ok_float v, ok_float rho) {
  return Abs(v) < 1 + 1 / rho ? v * rho / (1 + rho) : v - Sign(v) / rho;
}

__DEVICE__ inline ok_float ProxIdentity(ok_float v, ok_float rho) {
  return v - 1 / rho;
}

__DEVICE__ inline ok_float ProxIndBox01(ok_float v, ok_float rho) {
  return v <= 0 ? 0 : v >= 1 ? 1 : v;
}

__DEVICE__ inline ok_float ProxIndEq0(ok_float v, ok_float rho) {
  return 0;
}

__DEVICE__ inline ok_float ProxIndGe0(ok_float v, ok_float rho) {
  return v <= 0 ? 0 : v;
}

__DEVICE__ inline ok_float ProxIndLe0(ok_float v, ok_float rho) {
  return v >= 0 ? 0 : v;
}

__DEVICE__ inline ok_float ProxLogistic(ok_float v, ok_float rho) {
	ok_float x, l, inv_ex, f, g, g_rho;
	size_t i;

  	/* Initial guess based on piecewise approximation. */

	if (v < (ok_float)(-2.5))
		x = v;
	else if (v > (ok_float)(2.5) + 1 / rho)
		x = v - 1 / rho;
	else
		x = (rho * v - (ok_float)(0.5)) / ((ok_float)(0.2) + rho);

	/* Newton iteration. */
	l = v - 1 / rho, u = v;
	for (i = 0; i < 5; ++i) {
		inv_ex = 1 / (1 + Exp(-x));
		f = inv_ex + rho * (x - v);
		g = inv_ex * (1 - inv_ex) + rho;
		if (f < 0)
		  l = x;
		else
		  u = x;
		x = x - f / g;
		x = Min(x, u);
		x = Max(x, l);
	}

	/* Guarded method if not converged. */
	for (i = 0; u - l > MACHINETOL && i < 100; ++i) {
		g_rho = 1 / (rho * (1 + Exp(-x))) + (x - v);
		if (g_rho > 0) {
			l = Max(l, x - g_rho);
			u = x;
		} else {
			u = Min(u, x - g_rho);
			l = x;
		}
		x = (u + l) / 2;
	}
	return x;
}

__DEVICE__ inline ok_float ProxMaxNeg0(ok_float v, ok_float rho) {
  ok_float z = v >= 0 ? v : 0;
	return v + 1 / rho <= 0 ? v + 1 / rho : z;
}

__DEVICE__ inline ok_float ProxMaxPos0(ok_float v, ok_float rho) {
  ok_float z = v <= 0 ? v : 0;
	return v >= 1 / rho ? v - 1 / rho : z;
}

__DEVICE__ inline ok_float ProxNegLog(ok_float v, ok_float rho) {
	return (v + Sqrt(v * v + 4 / rho)) / 2;
}

__DEVICE__ inline ok_float ProxRecipr(ok_float v, ok_float rho) {
  v = Max(v, (ok_float) 0);
	return CubicSolve(-v, (ok_float) 0, -1 / rho);
}

__DEVICE__ inline ok_float ProxSquare(ok_float v, ok_float rho) {
	return rho * v / (1 + rho);
}

__DEVICE__ inline ok_float ProxZero(ok_float v, ok_float rho) {
	return v;
}

// Evaluates the proximal operator of f.
__DEVICE__ inline ok_float ProxEval(const FunctionObj *f_obj, 
									ok_float v, ok_float rho) {
	const ok_float a = f_obj.a, b = f_obj.b, c = f_obj.c, 
				   d = f_obj.d, e = f_obj.e;

	v = a * (v * rho - d) / (e + rho) - b;
	rho = (e + rho) / (c * a * a);

	if (f_obj->h == FnAbs) v = ProxAbs(v, rho);
	else if (f_obj->h == FnNegEntr) v = ProxNegEntr(v, rho);
	else if (f_obj->h == FnExp) v = ProxExp(v, rho);
	else if (f_obj->h == FnHuber) v = ProxHuber(v, rho);
	else if (f_obj->h == FnIdentity) v = ProxIdentity(v, rho);
	else if (f_obj->h == FnIndBox01) v = ProxIndBox01(v, rho);
	else if (f_obj->h == FnIndEq0) v = ProxIndEq0(v, rho);
	else if (f_obj->h == FnIndGe0) v = ProxIndGe0(v, rho);
	else if (f_obj->h == FnIndLe0) v = ProxIndLe0(v, rho);
	else if (f_obj->h == FnLogistic) v = ProxLogistic(v, rho);
	else if (f_obj->h == FnMaxNeg0) v = ProxMaxNeg0(v, rho);
	else if (f_obj->h == FnMaxPos0) v = ProxMaxPos0(v, rho);
	else if (f_obj->h == FnRecipr) v = ProxRecipr(v, rho);
	else if (f_obj->h == FnSquare) v = ProxSquare(v, rho);
	else v = ProxZero(v, rho);

	return (v + b) / a;
}



/*
// Function definitions.
//
// Each of the following functions corresponds to one of the Function enums.
// All functions accept one argument x and four parameters (a, b, c, and d)
// and returns the evaluation of
//
//   x -> c * f(a * x - b) + d * x.
*/
__DEVICE__ inline ok_float FuncAbs(ok_float x) {
  return Abs(x);
}

__DEVICE__ inline ok_float FuncNegEntr(ok_float x) {
  return x <= 0 ? 0 : x * Log(x);
}

__DEVICE__ inline ok_float FuncExp(ok_float x) {
  return Exp(x);
}

__DEVICE__ inline ok_float FuncHuber(ok_float x) {
  ok_float xabs = Abs(x);
  ok_float xabs2 = xabs * xabs;
  return xabs < (ok_float) 1 ? xabs2 / 2 : xabs - (ok_float)(0.5);
}

__DEVICE__ inline ok_float FuncIdentity(ok_float x) {
  return x;
}

__DEVICE__ inline ok_float FuncIndBox01(ok_float x) {
  return 0;
}

__DEVICE__ inline ok_float FuncIndEq0(ok_float x) {
  return 0;
}

__DEVICE__ inline ok_float FuncIndGe0(ok_float x) {
  return 0;
}

__DEVICE__ inline ok_float FuncIndLe0(ok_float x) {
  return 0;
}

__DEVICE__ inline ok_float FuncLogistic(ok_float x) {
  return Log(1 + Exp(x));
}

__DEVICE__ inline ok_float FuncMaxNeg0(ok_float x) {
  return MaxNeg(x);
}

__DEVICE__ inline ok_float FuncMaxPos0(ok_float x) {
  return MaxPos(x);
}

__DEVICE__ inline ok_float FuncNegLog(ok_float x) {
  x = Max((ok_float) 0, x);
  return -Log(x);
}

__DEVICE__ inline ok_float FuncRecpr(ok_float x) {
  x = Max((ok_float) 0, x);
  return 1 / x;
}

__DEVICE__ inline ok_float FuncSquare(ok_float x) {
  return x * x / 2;
}

__DEVICE__ inline ok_float FuncZero(ok_float x) {
  return 0;
}

/* Evaluates the function f. */
__DEVICE__ inline ok_float FuncEval(const FunctionObj *f_obj, ok_float x) {
	ok_float dx = f_obj.d * x;
	ok_float ex = f_obj.e * x * x / 2;
	x = f_obj.a * x - f_obj.b;


	if (f_obj->h == FnAbs) FuncAbs(x);
	else if (f_obj->h == FnNegEntr) FuncNegEntr(x);
	else if (f_obj->h == FnExp) FuncExp(x);
	else if (f_obj->h == FnHuber) FuncHuber(x);
	else if (f_obj->h == FnIdentity) FuncIdentity(x);
	else if (f_obj->h == FnIndBox01) FuncIndBox01(x);
	else if (f_obj->h == FnIndEq0) FuncIndEq0(x);
	else if (f_obj->h == FnIndGe0) FuncIndGe0(x);
	else if (f_obj->h == FnIndLe0) FuncIndLe0(x);
	else if (f_obj->h == FnLogistic) FuncLogistic(x);
	else if (f_obj->h == FnMaxNeg0) FuncMaxNeg0(x);
	else if (f_obj->h == FnMaxPos0) FuncMaxPos0(x);
	else if (f_obj->h == FnRecipr) FuncRecipr(x);
	else if (f_obj->h == FnSquare) FuncSquare(x);
	else v = FuncZero(v, rho);

	return f_obj->c * x + dx * ex;
}


void ProxEvalVector(const FunctionVector * f, ok_float rho,
			  const ok_float * x_in, size_t stride_in,
			  ok_float * x_out, size_t stride_out);

ok_float FuncEvalVector(const FunctionVector * f, const ok_float * x_in,
          		  size_t stride);


#ifdef
}		/* extern "C" */
#endif


#endif /* OPTKIT_PROXLIB_H_GUARD */