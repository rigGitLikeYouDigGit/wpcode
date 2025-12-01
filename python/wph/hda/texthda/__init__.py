from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


"""not a great idea to link this python package with the hda in houdini itself,
once we know what the text system needs, try to break it out


- generate final node contents by layering inputs
	- nodes, then connections, then params
- generate final diff by comparing 3 elements against incoming

save and display all of these

"""



