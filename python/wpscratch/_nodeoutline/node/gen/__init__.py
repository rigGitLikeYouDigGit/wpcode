

from __future__ import annotations
import typing as T


if T.TYPE_CHECKING:
	from .dag import Dag
	from .transform import Transform


	class Catalogue:
		Dag = Dag
		Transform = Transform


