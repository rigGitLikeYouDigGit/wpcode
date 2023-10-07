
from __future__ import annotations

"""objects that work with frames - 
these are between local variables and global,
letting only parts of a program inherit state from a
higher frame, without polluting the global environment.

Later this will also make code more robust to multithreading

allow embedding of dictionaries in frames, so that
calling code can look up through frames to find environment overrides

the word 'override' is losing all meaning, find something better

to state the obvious, don't use this if a better way is available
"""

import inspect

from wptree import Tree

# globals key in frames to use
FRAME_STANCHION_KEY = "__frame_stanchion__"

def getFrameStanchion(key, frame=None):
	"""get a key from a frame stanchion,
	returns earliest override from calling frame above this one
	"""
	if frame is None:
		frame = inspect.currentframe().f_back

	result = frame.f_locals.get(FRAME_STANCHION_KEY, {}).get(key)
	if result:
		return result
	if not frame.f_back:
		return None
	return getFrameStanchion(key, frame.f_back)

def setFrameStanchion(key, value, frame=None):
	"""set a key in a frame stanchion
	"""
	if frame is None:
		frame = inspect.currentframe().f_back

	stanchion = frame.f_locals.setdefault(FRAME_STANCHION_KEY, {})
	stanchion[key] = value

def popFrameStanchion(key, frame=None):
	"""pop a key from a frame stanchion
	"""
	if frame is None:
		frame = inspect.currentframe().f_back

	stanchion = frame.f_locals.setdefault(FRAME_STANCHION_KEY, {})
	return stanchion.pop(key)



