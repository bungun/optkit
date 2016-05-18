#include "optkit_pogs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

int private_api_accessible()
{
#ifdef OK_DEBUG_PYTHON
	return 1;
#else
	return 0;
#endif
}

ok_status set_default_settings(pogs_settings * s)
{
	OK_CHECK_PTR(s);
	s->alpha = kALPHA;
	s->rho = kOne;
	s->abstol = kATOL;
	s->reltol = kRTOL;
	s->maxiter = kMAXITER;
	s->verbose = kVERBOSE;
	s->suppress = kSUPPRESS;
	s->adaptiverho = kADAPTIVE;
	s->gapstop = kGAPSTOP;
	s->warmstart = kWARMSTART;
	s->resume = kRESUME;
	s->x0 = OK_NULL;
	s->nu0 = OK_NULL;
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
