
from __future__ import annotations
import typing as T

from wplib import log
from wptree import Tree, TreeInterface

from wpdex import WpDex, WpDexProxy, DexPathable


class TreeDex(WpDex):

	forTypes = [Tree, TreeInterface]
	dispatchInit = True

	# don't cover aux data for now
	mutatingMethodNames = {
		"setName",
		"setValue",
		#"setParent",
		"__call__",
		"addChild",
		"setIndex",
		"__setitem__",
		#"__getitem__",
		#"value",
	}

	obj : Tree

	def _buildChildren(self) ->dict[DexPathable.keyT, WpDex]:
		# don't do a single wpdex for the whole tree
		# for i in self.obj.allBranches(includeSelf=False):
		# 	self.makeChildPathable((i.,), i)
		items = {}
		for i in self.obj.branches:
			items[i.name] = self.makeChildPathable((i.name, ), i)
		items["name"] = self.makeChildPathable(("name",), self.obj.name)
		if self.obj.value is not None:
			items["value"] = self.makeChildPathable(("value",), self.obj.value)
		#log("buildChildren items", items)
		return items

if __name__ == '__main__':
	t = Tree("root")
	t["a"]
	t["a", "b"]

	proxy = WpDexProxy(t)
	proxy.dex().getEventSignal("main", create=True).connect(
		lambda *args: print("EVENT", args))
	#proxy.setValue(2)
	# b = proxy("a")
	# b.setValue(3)
	#

	b = proxy("a", "b")
	b.setValue(4)

	proxy["a", "b", "c", "d", "e"] = 4

	# branch = proxy("a", "b")
	# print("branch", branch, type(branch))
	# # # connect listener
	# dex = proxy.dex()

	# print("proxy", proxy, type(proxy))
	# print("proxy['a']", proxy["a"], type(proxy("a")))
	#proxy["a", "b", "c"] = 4
	#proxy("a", "b", "c").setValue( 6 )



