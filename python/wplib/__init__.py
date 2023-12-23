

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

# get root of WP installation
import sys, os
from pathlib import Path
WP_ROOT = os.getenv("WEPRESENT_ROOT")
assert WP_ROOT, "WEPRESENT_ROOT not set in environment"
WP_ROOT_PATH = Path(WP_ROOT)
