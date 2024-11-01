

"""dedicated library package for pure-python utilities -
nothing project-specific, nothing applied

This should depend on nothing else in wp, or any other project.

Will migrate things from tree.lib.object as needed, if they get used.
"""
from .log import log

from .coderef import CodeRef

#from .expression import *

from wplib.object import Adaptor, TypeNamespace, Sentinel

from .coerce import coerce

# get root of WP installation if found
# if you only want to use wplib and not the vfx packages, this can be ignored
import sys, os
from pathlib import Path
WP_ROOT = os.getenv("WEPRESENT_ROOT")
try:
	assert WP_ROOT, "WEPRESENT_ROOT not set in environment"
	WP_ROOT_PATH = Path(WP_ROOT)
except AssertionError:
	# fall back to checking up the folder tree to find a file named "WEPRESENT_ROOT"
	_thisPath = Path(__file__).parent

	while not (_thisPath / "WEPRESENT_ROOT").exists():
		if not _thisPath.parent: raise RuntimeError
		if _thisPath.parent == _thisPath: raise RuntimeError
		_thisPath = _thisPath.parent
	WP_ROOT_PATH = _thisPath

# set code path constants
WP_CODE_ROOT = WP_ROOT_PATH / "code"
WP_PY_ROOT = WP_CODE_ROOT / "python"
WP_PY_RESOURCE_PATH = Path(__file__).parent / "resource"

# extra convenience imports
from .pathable import Pathable
