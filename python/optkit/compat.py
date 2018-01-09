from __future__ import print_function
import six
import os

from six import add_metaclass

def listmap(f, *args):
    return list(six.moves.map(f, *args))
def listfilter(f, *args):
    return list(six.moves.filter(f, *args))

if six.PY3:
    from six.moves import xrange
    from six.moves import reduce
