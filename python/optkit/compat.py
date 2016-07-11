from __future__ import print_function

import sys
if sys.version_info.major > 2:
	xrange = range
	def listmap(f, *args):
		return list(map(f, *args))
	def listfilter(f, *args):
		return list(filter(f, *args))
else:
	listmap = map
	listfilter = filter