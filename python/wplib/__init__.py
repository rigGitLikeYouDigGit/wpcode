

"""dedicated library package for pure-python utilities -
nothing project-specific, nothing applied

This should depend on nothing else in wp, or any other project.

Will migrate things from tree.lib.object as needed, if they get used.
"""
from .log import log

from .coderef import CodeRef

#from .expression import *

from wplib.object import TypeNamespace, Sentinel



from .coerce import coerce