
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
"""

from .scene import WpCanvasScene
from .view import WpCanvasView
from .element import WpCanvasItem

