from __future__ import annotations
import types, typing as T
import pprint
import sys, os
from pathlib import Path
"""
TEMPLATE FILE for DCC startup commands - 
on call, file is copied and formatted for specific process.
passed to DCC on startup -
 
initialise any env variables

add the code root of WP_ROOT to python path-
TODO: make control / send this by idem, maybe - 
should be bootstrapping the maya session with all the paths and
environment we need.
"""

ARGS = $ARGS
KWARGS = $KWARGS

print("RUN WPM STARTUP")

WP_ROOT = os.getenv("WEPRESENT_ROOT")
assert WP_ROOT, "WEPRESENT_ROOT not set in environment"
WP_ROOT_PATH = Path(WP_ROOT)
WP_PY_ROOT = WP_ROOT_PATH / "code" / "python"
sys.path.insert(0, str(WP_PY_ROOT))
