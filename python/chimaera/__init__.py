
"""rebuild of original chimaera graph system -
dependency graph that can extend and modify itself

Each Chimaera node's value is an expression, and each node's connections
are locally scoped to that node.

Use node incoming connections and refmap to show dependency order in graph -
can't stop you circumventing it during node compute, but if you do, you're
on your own.

"""


from .node import ChimaeraNode, NodeAttrWrapper

