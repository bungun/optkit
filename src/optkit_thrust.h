#include "optkit_dense.h"
#include "optkit_defs_gpu.h"


/* thrust:: methods */
// x -> |x|
struct AbsF : thrust::unary_function<ok_float, ok_float> {
  __device__ ok_float operator()(ok_float x) { return MATH(fabs)(x); }
};

// x -> alpha / x
struct ReciprF : thrust::unary_function<ok_float, ok_float> {
  ok_float alpha;
  ReciprF() : alpha(1) {}
  ReciprF(ok_float alpha) : alpha(alpha) {}
  __device__ ok_float operator()(ok_float x) { return alpha / x; }
};

// x -> sqrt(x)
struct SqrtF : thrust::unary_function<ok_float, ok_float> {
  __device__ ok_float operator()(ok_float x) { return MATH(sqrt)(x); }
};

// x -> x^p
struct PowF : thrust::unary_function<ok_float, const ok_float> {
  const ok_float p;
  PowF(const ok_float p) : p(p) {} 
  __device__ ok_float operator()(ok_float x) { return MATH(pow)(x, p); }
};

strided_range_t
__make_strided_range(vector * v){
  return strided_range_t(
    thrust::device_pointer_cast(v->data),
    thrust::device_pointer_cast(v->data + v->stride * v->size), v->stride);
}

strided_range_t
__make_const_strided_range(const vector * v){
  return strided_range_t(
    thrust::device_pointer_cast(v->data),
    thrust::device_pointer_cast(v->data + v->stride * v->size), v->stride);
}

template <typename UnaryFunction>
void
__transform_r(strided_range_t r, UnaryFunction f){
  thrust::transform(r.begin(), r.end(), r.begin(), f);
}

template <typename BinaryFunction>
void 
__transform_rr(strided_range_t r1, strided_range_t r2, BinaryFunction f){
  thrust::transform(r1.begin(), r1.end(), r2.begin(), 
    r1.begin(), f);
}

template <typename BinaryFunction>
void
__transform_rc(strided_range_t r, ok_float x, BinaryFunction f){
  thrust::transform(r.begin(), r.end(), constant_iterator_t(x),
    r.begin(), f);
}

template <typename BinaryFunction>
void
__transform_cr(ok_float x, strided_range_t r, BinaryFunction f){
  thrust::transform(r.begin(), r.end(), constant_iterator_t(x),
    r.begin(), f);
}

void 
__thrust_vector_scale(vector * v, ok_float x) {
  strided_range_t r = __make_strided_range(v);
  __transform_rc(r, x, thrust::multiplies<ok_float>());
}

void 
__thrust_vector_add(vector * v1, const vector * v2) {
  strided_range_t r1 = __make_strided_range(v1);
  strided_range_t r2 = __make_const_strided_range(v2);
  __transform_rr(r1, r2, thrust::plus<ok_float>());
}

void 
__thrust_vector_sub(vector * v1, const vector * v2) {
  strided_range_t r1 = __make_strided_range(v1);
  strided_range_t r2 = __make_const_strided_range(v2);
  __transform_rr(r1, r2, thrust::minus<ok_float>());
}

void 
__thrust_vector_mul(vector * v1, const vector * v2) {
  strided_range_t r1 = __make_strided_range(v1);
  strided_range_t r2 = __make_const_strided_range(v2);
  __transform_rr(r1, r2, thrust::multiplies<ok_float>());
}

void 
__thrust_vector_div(vector * v1, const vector * v2) {
  strided_range_t r1 = __make_strided_range(v1);
  strided_range_t r2 = __make_const_strided_range(v2);
  __transform_rr(r1, r2, thrust::divides<ok_float>());
}

void 
__thrust_vector_add_constant(vector * v, const ok_float x) {
  strided_range_t r = __make_strided_range(v);
  __transform_rc(r, x, thrust::plus<ok_float>());
}

void 
__thrust_vector_abs(vector * v) {
  strided_range_t r = __make_strided_range(v);
  __transform_r(r, AbsF());
}

void 
__thrust_vector_recip(vector * v) {
  strided_range_t r = __make_strided_range(v);
  __transform_r(r, ReciprF());
}
void 
__thrust_vector_sqrt(vector * v) {
  strided_range_t r = __make_strided_range(v);
  __transform_r(r, SqrtF());
}

void 
__thrust_vector_pow(vector * v, const ok_float p){
  strided_range_t r = __make_strided_range(v);
  __transform_r(r, PowF(p));  
}