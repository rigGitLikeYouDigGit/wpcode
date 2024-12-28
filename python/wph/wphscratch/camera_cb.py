from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


"""
hou.nodeEventType.ParmTupleChanged
^ this isn't ideal, fires on any tuple change, and fires separately for every
single param on the node, regardless of if they've changed
also some individual params can be duplicated
for cameras at least


but it seems like there isn't a more specific one,
so we just need to rate-limit the callback in the same way as
with maya
"""