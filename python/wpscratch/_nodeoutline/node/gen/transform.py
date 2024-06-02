
from __future__ import annotations
import typing as T

# from .dag import Dag as Dag

# import using catalogue to solve dependency loop
from .. import retriever

Dag = retriever.getNodeCls("Dag")

if T.TYPE_CHECKING:
	from .. import Dag

class Transform(Dag):

	pass