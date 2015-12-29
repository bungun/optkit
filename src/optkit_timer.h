#ifndef OPTKIT_TIMER_H_GUARD
#define OPTKIT_TIMER_H_GUARD

#include "optkit_defs.h"
#include <unistd.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C"{
#endif

ok_float OK_TIMER() {
  struct timeval tv;
  gettimeofday(&tv, OK_NULL);
  return (ok_float) tv.tv_sec + (ok_float) tv.tv_usec * (ok_float) 1e-6;
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_TIMER_H_GUARD */