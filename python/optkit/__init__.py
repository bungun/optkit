from optkit.compat import *

import os

if int(os.getenv('OPTKIT_C_TESTING', 0)) == 0:
	from optkit.api import OPTKIT_VERSION

	# Backend
	from optkit.api import set_backend
	if int(os.getenv('OPTKIT_IMPORT_BACKEND', 0)) > 1:
		from optkit.api import backend

	# C implementations
	from optkit.api import PogsSolver, PogsObjective
	from optkit.api import PogsAbstractSolver
	from optkit.api import Clustering, ClusteringSettings

	del utils
	del libs
	del types
	del backends
	# del api
