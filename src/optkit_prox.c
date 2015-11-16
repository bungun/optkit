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
		f->objectives[i] = (FunctionObj){FnZero,1,0,1,0,0}; 
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
			f->objectives[i] = (FunctionObj){FnZero,1,0,1,0,0};
	} else if (a == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){h[i],1,0,1,0,0};
	} else if (b == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){h[i],a[i],0,1,0,0};
	} else if (c == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){h[i],a[i],b[i],1,0,0};
	} else if (d == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){h[i],a[i],b[i],c[i],0,0};

	} else if (e == OK_NULL) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){h[i],a[i],b[i],c[i],d[i],0};
	} else {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = 0; i < f->size; ++i)
			f->objectives[i] = (FunctionObj){h[i],a[i],b[i],c[i],d[i],e[i]};
	}

}


void function_vector_print(FunctionVector * f){
	size_t i;
	for (i = 0; i < f->size; ++i)
		printf("h: %i, a: %0.2e, b: %0.2e, c: %0.2e, d: %0.2e, e: %0.2e\n", 
				(int) f->objectives[i].h, f->objectives[i].a,
				f->objectives[i].b, f->objectives[i].c,
				f->objectives[i].d, f->objectives[i].e);
}

void ProxEvalVector(const FunctionVector * f, ok_float rho, 
			  const vector * x_in, vector * x_out) {
	
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (uint i = 0; i < f->size; ++i){
		x_out->data[i * x_out->stride] = ProxEval(&f->objectives[i], 
			x_in->data[i * x_in->stride], rho);
	}
}


ok_float FuncEvalVector(const FunctionVector * f, const vector * x) {
	ok_float sum = 0;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(+:sum)
	#endif
	for (uint i = 0; i < f->size; ++i)
		sum += FuncEval(&f->objectives[i], x->data[i * x->stride]);
	return sum;
}


#ifdef __cplusplus
}		/* extern "C" */
#endif