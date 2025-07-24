

from wpm import om

"""ON STRUCTURE:
lib can depend on WN if needed - 
WN wrapper classes later can import lib functions dynamically,
so we should be clear of loops
"""

def execPythonStringOnIdle(s:str):
	"""wrapper for MGlobal.executeCommandOnIdle -
	shunts a python string command through mel.

	still no idea how stack frames work with this,
	might need to pull the variables of the calling
	frame into this function
	"""
	om.MGlobal.executeCommandOnIdle(f"python( \"{s}\");")
