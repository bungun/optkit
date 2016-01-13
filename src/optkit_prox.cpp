 #include "optkit_prox.hpp"

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
	f->objectives = (FunctionObj *) h;
}

void function_vector_memcpy_va(FunctionVector * f, FunctionObj * h){
	memcpy(f->objectives, h, f->size * sizeof(FunctionObj));
}

void function_vector_memcpy_av(FunctionObj * h, FunctionVector * f){
	memcpy(h, f->objectives, f->size * sizeof(FunctionObj));
}


void 
function_vector_mul(FunctionVector * f, const vector * v){
	size_t i;
	vector el = (vector){f->size, 
		sizeof(FunctionObj) / sizeof(ok_float), OK_NULL};

	el.data = &(f->objectives->a);
	for (i = 0; i < f->size; ++i)
		el.data[i * el.stride] *= v->data[i * v->stride];

	el.data = &(f->objectives->d);
	for (i = 0; i < f->size; ++i)
		el.data[i * el.stride] *= v->data[i * v->stride];

	el.data = &(f->objectives->e);
	for (i = 0; i < f->size; ++i)
		el.data[i * el.stride] *= v->data[i * v->stride];
}

void 
function_vector_div(FunctionVector * f, const vector * v){
	size_t i;
	vector el = (vector){f->size, 
		sizeof(FunctionObj) / sizeof(ok_float), OK_NULL};

	el.data = &(f->objectives->a);
	for (i = 0; i < f->size; ++i)
		el.data[i * el.stride] /= v->data[i * v->stride];

	el.data = &(f->objectives->d);
	for (i = 0; i < f->size; ++i)
		el.data[i * el.stride] /= v->data[i * v->stride];

	el.data = &(f->objectives->e);
	for (i = 0; i < f->size; ++i)
		el.data[i * el.stride] /= v->data[i * v->stride];
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
	uint i;	
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < f->size; ++i){
		x_out->data[i * x_out->stride] = ProxEval(&f->objectives[i], 
			x_in->data[i * x_in->stride], rho);
	}
}


ok_float FuncEvalVector(const FunctionVector * f, const vector * x) {
	ok_float sum = 0;
	uint i;
	#ifdef _OPENMP
	#pragma omp parallel for reduction(+:sum)
	#endif
	for (i = 0; i < f->size; ++i)
		sum += FuncEval(&f->objectives[i], x->data[i * x->stride]);
	return sum;
}


#ifdef __cplusplus
}		/* extern "C" */
#endif
