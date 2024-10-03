
from __future__ import annotations
import typing as T

from dataclasses import dataclass, asdict, fields, is_dataclass

from wplib import log
from wpdex.base import WpDex

from typing import ClassVar, Dict, Protocol, Any, TypedDict


class IsDataclass(Protocol):
	# as already noted in comments, checking for this attribute is currently
	# the most reliable way to ascertain that something is a dataclass
	__dataclass_fields__: ClassVar[Dict[str, Any]]

@dataclass
class Foo:
	pass

class DataClassDex(WpDex):
	"""dex for dataclass - seems like we can just use the default
	visitor for it"""
	forTypes = (is_dataclass, )




if __name__ == '__main__':

	@dataclass
	class MyTest:
		name : str

	myTest = MyTest("eyyy")

	print(myTest)
	dex = WpDex(myTest)
	print(type(dex))
	print(dex)

