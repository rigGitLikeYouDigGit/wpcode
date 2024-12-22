from __future__ import annotations
import types, typing as T
import pprint
import sys, os
sys.path.append("C:\Python39\Lib\site-packages")

from pathlib import Path


"""
maya-specific, Idem-specific setup

this should run after the normal pipeline startup file?
but for now we just run this

add the code root of WP_ROOT to python path-
TODO: make control / send this by idem, maybe - 
should be bootstrapping the maya session with all the paths and
environment we need.
"""



print("RUN WPM STARTUP")

WP_ROOT = os.getenv("WEPRESENT_ROOT")
assert WP_ROOT, "WEPRESENT_ROOT not set in environment"
WP_ROOT_PATH = Path(WP_ROOT)
WP_PY_ROOT = WP_ROOT_PATH / "code" / "python"
sys.path.insert(0, str(WP_PY_ROOT))


from wplib import log
from idem.dcc import maya as idmaya
dccType = idmaya.MayaProcess
import orjson

# check if any idem parametres packed in system args
idemParams = None
for i in sys.argv:
	if i.startswith("idemParams::"):
		idemParams = orjson.loads(i.split("idemParams::", 1)[1])
		break

log("idemParams", idemParams, type(idemParams))



