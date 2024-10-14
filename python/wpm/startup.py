
from __future__ import  annotations
import typing as T
"""single file to be run on maya startup -
we no longer truster userSetup.py in the maya folder, 
this file will be run directly when launched from idem
or any other way in the WP pipeline.

add the code root of WP_ROOT to python path-
TODO: make control / send this by idem, maybe - 
should be bootstrapping the maya session with all the paths and
environment we need.

could generate a templated temp startup file to get around
the command packing on the command line directly, I did that
at rocksteady and it worked pretty well 

maybe it's fine to copy/paste these lines
"""
print("RUN WPM STARTUP")
import sys, os
from pathlib import Path
WP_ROOT = os.getenv("WEPRESENT_ROOT")
assert WP_ROOT, "WEPRESENT_ROOT not set in environment"
WP_ROOT_PATH = Path(WP_ROOT)
WP_PY_ROOT = WP_ROOT_PATH / "code" / "python"
sys.path.insert(0, str(WP_PY_ROOT))

"""I can immediately see how a proper package manager like Rez would be more helpful
here, in case we need to override specific versions of numpy, taichi etc"""