

from __future__ import annotations
import typing as T


#from ..gen import Catalogue

# class Catalogue(Catalogue):
# 	pass

if T.TYPE_CHECKING:
	from .dag import Dag
	from .transform import Transform


	class Catalogue:
		Dag = Dag
		Transform = Transform

