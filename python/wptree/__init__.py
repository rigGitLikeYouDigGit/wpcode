from __future__ import annotations
import typing as T
"""soft reboot of main tree - still not sure how best to structure it.
Eventually replace the normal tree with this one.

This package depends on wplib - others may depend on this.
No project-specific code here, and files should relate only to the tree system.

If tree ends up being a special case of a more general graph system, then
place the graph system in wplib, and have this package depend on it.

"""

from .interface import TreeInterface, TreeType
from .main import Tree
from .delta import TreeDeltaAid, TreeDeltas

from .dex import TreeDex

