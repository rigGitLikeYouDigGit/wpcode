
from __future__ import annotations

import pprint
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



# from param import rx
#
# class myrx(rx):
# 	def __str__(self):
# 		return "<rx" + str(self._obj) + ">"
# 	pass
if __name__ == '__main__':

	eventFn = lambda *args: (print("EVENT"), pprint.pprint(args[0]))
	t = Tree("root")
	#t["a"]

	#print(t.branches)
	#raise
	# t["a", "b"]
	#t["a", "b", "c"] = 4

	p = WpDexProxy(t)
	print(p.dex().children())
	p.dex().getEventSignal().connect(eventFn)
	p.name = "test"

	# log("before call")
	# t = p("a")




	# p.name = "test"


	# log("-----", p, type(p), )
	# log("px call fn", p.__call__)
	# print("")
	# a = p("a")
	# log("-----", a, type(a))
	# c1 = a("b", "c")
	# c2 = a("b", "c")
	# log(c2, type(c2), c1 is c2)
	#
	# c2.value = [1, [2]]
	# c2.setValue( [1, [2]] )
	# log(c2.value[1][0], type(c2.value[1][0]))
	# log(type(c2.value).__mro__)
	#
	# log(c2.value._proxyData)
	# log(type(c2.value._proxyData["parent"]))


	#c2.value = "hello"



	#b = t("a", "b", "c")

	# proxy = WpDexProxy(t)
	# proxy.dex().getEventSignal("main", create=True).connect(eventFn)
	# #proxy.setValue(2)
	# # b = proxy("a")
	# # b.setValue(3)
	# #
	# #proxy["a", "b", "c", "d", "e"] = 4
	# #b = proxy("a", "b")
	# #b.setValue(4)
	#
	# #proxy["a", "b", "c", "d", "e"] = 4
	# proxy["a", "b", "c"] = 7
	# print("jjj")
	# proxy["a", "b", "c"] = 9
	#
	# # branch = proxy("a", "b")
	# # print("branch", branch, type(branch))
	# # # # connect listener
	# # dex = proxy.dex()
	#
	# # print("proxy", proxy, type(proxy))
	# # print("proxy['a']", proxy["a"], type(proxy("a")))
	# #proxy["a", "b", "c"] = 4
	# #proxy("a", "b", "c").setValue( 6 )



