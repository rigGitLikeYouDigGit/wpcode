
from __future__ import annotations
import typing as T

import networkx as nx

from wplib import log

from wptree import Tree, TreeInterface

from chimaera import ChimaeraNode, NodeType

"""experiments in chimaera graph execution - 
rich structure of the graph itself is already captured in
nodes - 
here we need to know what to execute if a given node has
to resolve.


individual tree branches can be nodes in the graph,
based on their inputs and connections - 

a tree is marked clean if it and all its branches are clean

"""


