

from __future__ import  annotations
"""file holding different typed events for tree system
these are intended to be thin, contained data can be anything of
any format

"""
from dataclasses import dataclass
from tree.lib.object import EventBase




@dataclass
class TreeEvent(EventBase):
	"""base class for trees -
	data can any object"""
	data : object = ()
	sender : object = None
	#accepted : bool = False

