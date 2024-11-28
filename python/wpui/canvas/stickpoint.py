from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets

"""
mixin to stick/suggest when dragging mouse 
events - 
unsure how to integrate this with a "real" component,
since connection might be implicit, might just
send an event back to change connection within deep state, etc

for full feature of connecting, need a lot more logic 
around suggested spots, highlighted spots to drag from,
etc

connectionPoint might expand with number of connections drawn,
might remember its own number of connections, or shrink to
only real entries
connected pipes should highlight tips when mouse is close, showing
they can be disconnected by dragging away

"""
