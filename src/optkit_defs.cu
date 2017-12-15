#include "optkit_defs.h"
#include "optkit_defs_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void optkit_version(int *maj, int *min, int *change, int *status) {
	*maj = OPTKIT_VERSION_MAJOR;
	*min = OPTKIT_VERSION_MINOR;
	*change = OPTKIT_VERSION_CHANGE;
	*status = (int) OPTKIT_VERSION_STATUS;
}

ok_status ok_device_reset()
{
	cudaDeviceReset();
	return OK_STATUS_CUDA;
}

#ifdef __cplusplus
}
#endif

