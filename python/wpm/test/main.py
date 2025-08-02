from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


from wpm import MAYA_PY_ROOT
import unittest
loader = unittest.TestLoader()
startDir = MAYA_PY_ROOT / "test"

def runTests():
	suite = loader.discover(startDir)
	runner = unittest.TextTestRunner()
	runner.run(suite)


