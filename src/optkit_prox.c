#include "optkit_prox.h"

#ifdef __cplusplus
extern "C" {
#endif


void function_vector_alloc(FunctionVector * f, size_t n){
	if (f->objectives != OK_NULL) ok_free(f->objectives);
	f->size = n;
	f->objectives = (FunctionObj *) malloc( n * sizeof(FunctionObj) );
}

void function_vector_calloc(FunctionVector * f, size_t n){
	size_t i;

	function_vector_alloc(f, n);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < n; ++i)
		f->objectives[i] = (FunctionObj){1,0,1,0,0,FnZero}; 
}

void function_vector_free(FunctionVector * f){
	if (f->objectives != OK_NULL) ok_free(f->objectives);
	f->size = 0;	
}

void function_vector_view_array(FunctionVector * f, 
                                FunctionObj * h, size_t n){
	f->size=n;
	f->objectives= (FunctionObj *) h;
}

void function_vector_from_multiarray(FunctionVector * f, Function_t * h, 
									 ok_float * a, ok_float * b, 
									 ok_float * c, ok_float * d, 
									 ok_float * e, size_t n){
	f->size=n;

	function_vector_alloc(f, n);
	function_vector_memcpy_vmulti(f, h, a, b, c, d, e);
}


void function_vector_memcpy_va(FunctionVector * f, FunctionObj * h){
	memcpy(f->objectives, h, f->size * sizeof(FunctionObj));
}

void function_vector_memcpy_vmulti(FunctionVector * f, Function_t *h,
									 ok_float * a, ok_float * b, 
									 ok_float * c, ok_float * d, 
									 ok_float * e){

	size_t i;

	if (h == OK_NULL){
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){1,0,1,0,0,FnZero};
	} else if (a == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){1,0,1,0,0, h[i]};
	} else if (b == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){a[i],0,1,0,0,h[i]};
	} else if (c == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){a[i],b[i],1,0,0,h[i]};
	} else if (d == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){a[i],b[i],c[i],0,0,h[i]};

	} else if (e == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){a[i],b[i],c[i],d[i],0,h[i]};
	} else {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){a[i], b[i], c[i], 
											 d[i], e[i], h[i]};
	}

}


void ProxEvalVector(const FunctionVector * f, ok_float rho, 
			  const ok_float * x_in, size_t stride_in, 
			  ok_float * x_out, size_t stride_out) {
	
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (uint i = 0; i < f->size; ++i)
		x_out[i * stride_out] = ProxEval(&f->objectives[i], x_in[i * stride_in], rho);
}


ok_float FuncEvalVector(const FunctionVector * f, const ok_float * x_in, 
				  size_t stride) {
	ok_float sum = 0;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(+:sum)
	#endif
	for (uint i = 0; i < f->size; ++i)
		sum += FuncEval(&f->objectives[i], x_in[i * stride]);
	return sum;
}


#ifdef __cplusplus
}		/* extern "C" */
#endif