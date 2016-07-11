from __future__ import print_function

import sys
if sys.version_info.major > 2:
	xrange = range
	def listmap(f, *args):
		return list(map(f, *args))
else:
	listmap = map