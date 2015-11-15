#include "optkit_prox.h"


#ifdef __cplusplus
extern "C" {
#endif

void checkvexity(FunctionObj * f){
	if (f->c < (ok_float) 0){
		printf("WARNING: f not convex for c < 0
				(provided: %e). Using c = 0", f->c);	
		f->c = (ok_float) 0;	
	}
	if (f->e < (ok_float) 0){
		printf("WARNING: f not convex for e < 0
				(provided: %e). Using e = 0", f->e);	
		f->e = (ok_float) 0;	
	}
}

/*
void init_fobj(FunctionObj * f){
	f->a = 1;
	f->b = 0;
	f->c = 1;
	f->d = 0;
	f->e = 0;
	f->h = FnZero;
}

void init_fobj_a(FunctionObj * f, Function_t h, ok_float a){
	f->a = a;
	f->b = 0;
	f->c = 1;
	f->d = 0;
	f->e = 0;
	f->h = h;
}
void init_fobj_ab(FunctionObj * f, Function_t h, 
				  ok_float a, ok_float b){
	f->a = a;
	f->b = b;
	f->c = 1;
	f->d = 0;
	f->e = 0;
	f->h = h;
}
void init_fobj_ac(FunctionObj * f, Function_t h, 
				  ok_float a, ok_float b, ok_float c){
	f->a = a;
	f->b = b;
	f->c = c;
	f->d = 0;
	f->e = 0;
	f->h = h;
	checkvexity(f);
}

void init_fobj_ad(FunctionObj * f, Function_t h, 
				  ok_float a, ok_float b,
				  ok_float c, ok_float d){
	f->a = a;
	f->b = b;
	f->c = c;
	f->d = d;
	f->e = 0;
	f->h = h;
	checkvexity(f);
}
void init_fobj_ae(FunctionObj * f, Function_t h, 
				  ok_float a, ok_float b,
				  ok_float c, ok_float d,
				  ok_float e){
	f->a = a;
	f->b = b;
	f->c = c;
	f->d = d;
	f->e = e;
	f->h = h;
	checkvexity(h);
}
*/

void function_vector_alloc(FunctionVector * f, size_t n){
	if (f->objectives != OK_NULL) ok_free(f->objectives);
	f->size = len;
	f->objectives = (FunctionObj *) malloc( n * sizeof(FunctionObj) );
}

void function_vector_calloc(FunctionVector * f, size_t n){
	size_t i;

	function_vector_alloc(f, len)
	for (i = 0; i < n; ++i)
		f->objectives[i] = (FunctionObj){1,0,1,0,0,FnZero}; 
}

void function_vector_free(FunctionVector * f){
	if (f->objectives != OK_NULL) ok_free(f->objectives);
	f->size = 0;	
}


void ProxEval(const * FunctionVector f, ok_float rho, 
			  const ok_float * x_in, size_t stride_in, 
			  ok_float * x_out, size_t stride_out) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned int i = 0; i < f_obj.size(); ++i)
		x_out[i * stride_out] = ProxEval(f->objectives[i], 
    								     x_in[i * stride_in], rho);
}


ok_float FuncEval(const * FunctionVector f, const ok_float * x_in, size_t stride) {
	ok_float sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
	for (unsigned int i = 0; i < f_obj.size(); ++i)
		sum += FuncEval(f_obj[i], x_in[i * stride]);
	return sum;
}


#ifdef
}		/* extern "C" */
#endif