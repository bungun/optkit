#ifndef OPTKIT_TIMER_H_GUARD
#define OPTKIT_TIMER_H_GUARD

#include "optkit_defs.h"
#include <unistd.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C"{
#endif

typedef struct OK_TIMER{
	struct timeval tv;
	double tic, toc;
} OK_TIMER;

OK_TIMER tic(){
	struct timeval tv;
	OK_TIMER timer = (OK_TIMER){tv, 0, 0};
	gettimeofday(&(timer.tv), OK_NULL);
	timer.tic = (double) timer.tv.tv_sec + (double) timer.tv.tv_usec * 1e-6;
	return timer;
}

ok_float toc(OK_TIMER timer){
	gettimeofday(&(timer.tv), OK_NULL);
	timer.toc = (double) timer.tv.tv_sec + (double) timer.tv.tv_usec * 1e-6;
	return (ok_float) (timer.toc - timer.tic);
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_TIMER_H_GUARD */