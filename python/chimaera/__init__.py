
"""rebuild of original chimaera graph system -
dependency graph that can extend and modify itself

Each Chimaera node's value is an expression, and each node's connections
are locally scoped to that node.

Use node incoming connections and refmap to show dependency order in graph -
can't stop you circumventing it during node compute, but if you do, you're
on your own.

expand and collapse systems are analoguous to generators for procedural
data structure elements

QA: why use a "node", "attr", "path" triple for input attrs? why not just full paths?
- because we need to resolve the attribute tree before pathing into it
- could also somehow embed a call within a path to evaluate the tree, but
	for that we need to extend the pathing syntax, maybe having attributes as a tree subclass

"""


from .node import ChimaeraNode#, NodeAttrWrapper

from .dex import ChiDex, ChiDeltaAid
