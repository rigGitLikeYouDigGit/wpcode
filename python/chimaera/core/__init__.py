

"""
maybe we can separate the DATA and the EXECUTION of the graph - 

consider main ChimaeraGraph holding data and expressions, and a
DirtyGraph being built over them?
Still probably need the DirtyNode interface in the base class


if we give up the dream for now, and just make a rigging system,
we need something to compute a load of nodes in order.



"""


from .node import ChimaeraNode
from .plugnode import PlugNode
from .construct import NodeFnSet

from .data import DataTree