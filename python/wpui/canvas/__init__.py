
from __future__ import annotations
import typing as T
"""base for interactive uis where elements can be selected,
moved etc

emulate maya:
cursor colour
selected colour
unselected colour

after a long and lonely road we have a decent way of doing this using 
a core python datastructure as the "model", and everything else
updating from it

EMBEDDED WIDGETS
fundamentally, embedded widgets break the view/model (or view/scene) system,
since they're children of graphicsItems in the scene, not separate per graphicsView
(in this way, indexWidgets in itemViews are more correct)

"""

from .scene import WpCanvasScene
from .view import WpCanvasView
from .element import WpCanvasElement, WpCanvasProxyWidget
from .connection import ConnectionPoint, ConnectionGroupDelegate, ConnectionsPainter

"""
TODO:
	terrible idea incoming:
	when a view widget gains focus, set the PARENT of each top-level embedded widget
	to be that view, and mask/clip them as if they actually were embedded in the scene?
	that way ALL this event spaghetti would go away

"""

