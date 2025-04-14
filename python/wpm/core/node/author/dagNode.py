
from __future__ import annotations
import typing as T

from ..gen.dagNode import DagNode as GenDagNode


class DagNode(GenDagNode):
	clsIsDag = True
	pass
