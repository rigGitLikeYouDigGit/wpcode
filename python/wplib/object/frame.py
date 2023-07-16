
from __future__ import annotations

"""lib functions for inspecting stacks and call frames"""

import inspect, traceback, trace

def getFrames():

	inspect.currentframe()

def errorWhenFnNameInStack(fnName:str):
	"""raise error only when given function appears in stack
	for when a debugger isn't available"""



