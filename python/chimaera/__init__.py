
"""rebuild of original chimaera graph system -
dependency graph that can extend and modify itself

Each Chimaera node's value is an expression, and each node's connections
are locally scoped to that node.


Plug nodes operate one layer above - each plug is a basic chimaera node,
Plugs output complex data objects, which are contained as trees.

Selection of nodes may be collapsed to data object, operated on,
then expanded again.

All levels support filtering on what part of data stream should be worked on -
child plugs in compound effectively filter the data object of their parent plug.

When I first laid out this system I was dead set against nodes having a single concrete path - I have absolutely no idea why. For now, nodes are contained in graph scope hierarchy - each graph is also a node, as with the Houdini model.




"""

from .core import *
