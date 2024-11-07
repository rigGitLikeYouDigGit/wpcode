
from __future__ import annotations
import typing as T, types

"""lib for wpdex - doesn't stand for anything, just a name

this is our overall pathing / expression resolution system,
and associated ui tools. 

depends on wplib and wpui

should this also handle reactive updates, links to UI etc?
the dream for me is to have a single model across the entire
program, updating live whenever a change comes from code or ui - 
obviously that's the whole point of QT, but the default views
are not enough 

arch-enemy situation that will never be resolved:

>>>proxyTree = myNode.settings()
>>>myNonProxyTree = Tree(a, b, c)
>>>proxyTree.addBranch(myNonProxyTree)
OBVIOUSLY this reference to the non-proxy tree can't be changed


"""


from .base import WpDex


from .primdex import *
from .dictdex import *
from .seqdex import *
from .strexpdex import *
from .pathdex import *
from .dataclassdex import *

from .proxy import WpDexProxy, WX#, Reference

from .react import EVAL, EVALAK, BIND, WRAP_MEMBERS, rx

from .modelled import Modelled

def getWpDex(obj:(WpDexProxy, WpDex, WX, rx, T.Any))->WpDex:
	"""maybe this isn't necessary -
	type coercions to move between actual values,
	proxies, WpDex and rx values
	if no existing wpDex, return None, so caller can work out how to proceed
	"""
	if isinstance(obj, WpDex): return obj
	if isinstance(obj, WpDexProxy): return obj.dex()
	if isinstance(obj, WX): return obj._kwargs["_dex"]
	return None
