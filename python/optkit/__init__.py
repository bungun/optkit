from os import getenv

from optkit.api import OPTKIT_VERSION

# Backend
from optkit.api import set_backend
if getenv('OPTKIT_IMPORT_BACKEND', 0) > 1:
	from optkit.api import backend

# C implementations
from optkit.api import PogsSolver, PogsObjective

del utils
del libs
del types
del backends
del api
del getenv