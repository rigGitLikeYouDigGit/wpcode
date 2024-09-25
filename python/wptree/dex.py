
from __future__ import annotations

import pprint
import typing as T

from wplib import log
from wplib.delta import DeltaAtom
from wptree import Tree, TreeInterface, TreeDeltaAid, TreeDeltas

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

	# def _buildChildren(self) ->dict[DexPathable.keyT, WpDex]:
	# 	# don't do a single wpdex for the whole tree
	# 	# for i in self.obj.allBranches(includeSelf=False):
	# 	# 	self.makeChildPathable((i.,), i)
	# 	items = {}
	# 	for i in self.obj.branches:
	# 		items[i.name] = self.makeChildPathable((i.name, ), i)
	# 	items["name"] = self.makeChildPathable(("name",), self.obj.name)
	# 	if self.obj.value is not None:
	# 		items["value"] = self.makeChildPathable(("value",), self.obj.value)
	# 	#log("buildChildren items", items)
	# 	return items

	def compareState(self, newDex:WpDex, baseDex:WpDex=None) ->(dict, list[DeltaAtom]):
		"""trees should recurse down into values for everything other than
		adding/removing branches and changing index"""
		deltas = super().compareState(newDex=newDex, baseDex=baseDex)
		deltas = [i for i in deltas if not isinstance(i,
                  (TreeDeltas.Name, TreeDeltas.Value))]
		return deltas

	def _consumeFirstPathTokens(self, path:pathT) ->tuple[list[WpDex], pathT]:
		"""process a path token"""
		path = tuple(path)
		#log("consume first tokens", self, path, path in self.keyDexMap, self.keyDexMap)
		token, *path = path
		# if isinstance(token, int):
		# 	return [self.children[token]], path
		return [self.keyDexMap[(token, )]], path



# from param import rx
#
# class myrx(rx):
# 	def __str__(self):
# 		return "<rx" + str(self._obj) + ">"
# 	pass
if __name__ == '__main__':

	# from param import rx
	# v = 3
	# rv = rx(v)
	# rv.rx.watch(lambda x : print("output", x))
	# v += 1
	# #rv += 1 # this invalidates rv apparently
	# rv.rx.value += 1
	"""raw rx really does seem a non-starter"""



	eventFn = lambda *args: (print("EVENT"), pprint.pprint(args[0]))
	t = Tree("root")
	#t["a"]

	#print(t.branches)
	#raise
	# t["a", "b"]
	#t["a", "b", "c"] = 4

	t("a").value = 33
	dex = WpDex(t)

	a = dex.staticCopy()
	b = dex.staticCopy()
	assert a.obj.uid == b.obj.uid

	log(WpDex.dexForObj(t))

	p = WpDexProxy(t)
	p.dex().getEventSignal().connect(eventFn)


	log(p("a") is dex.access(dex, "a", values=False).obj)
	#log(p("a"))

	log("fffff")
	# log(id(t) in dex.objIdDexMap)
	# log(id(t("a")) in dex.objIdDexMap)
	# log(dex.dexForObj(t("a")))

	a = p("a")
	log(a, a.root) # both proxies
	d = a("b", "c", "d")
	log("###############")
	#log(d)
	log(d.dex())

	f = a("b", "gg", "f")
	print("")
	log("##########", f)
	cpar = f.commonParent(d)
	log(cpar)


	#print(dex.allBranches(includeSelf=1))


	# p = WpDexProxy(t)
	# print(p.dex().children())
	# p.dex().getEventSignal().connect(eventFn)
	# #print("BEFORE SET NAME ###########")
	# #p.name = "test"
	# # print(p.dex().branches)
	# # print("##################")
	# # p.value = 33
	# # print("##################")
	# # p.value = 33
	# # print("##################")
	# # p.value = 55
	# # print("##########")
	# # log("before call")
	# # t = p("a")
	# print("#########")
	# t = p("a")
	#
	# log(t.commonParent(p))
	# log(t, type(t))



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



