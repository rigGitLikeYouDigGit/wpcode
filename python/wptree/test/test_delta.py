
from __future__ import print_function

import unittest

from tree import Tree, Proxy
from tree.dev.delta import ListDeltaProxy


def makeTestObjects():
	tempTree = Tree(name="testRoot", val="tree root")
	tempTree("branchA").value = "first branch"
	tempTree("branchA")("leafA").value = "first leaf"
	tempTree("branchB").value = 2

	baseDict = {"a" : "b", 5 : 65}

	baseList = ["a", "b", "c", "d"]

	return {"tree" : tempTree, "baseDict" : baseDict,
	        "list" : baseList
	        }

class DebugProxy(Proxy):

	# def __init__(self, obj):
	# 	print("super {}".format(super(DebugProxy, self)))
	# 	super(DebugProxy, self).__init__(obj)

	def __eq__(self, other):
		# print("super {}".format(super(DebugProxy, self)))
		#result = super(DebugProxy, self).__eq__(other)

		print("other", other, type(other))
		result = self._proxyObj.__eq__(other)
		# does this work
		#result = other.__eq__(self._proxyObj)
		print("debug eq {} == {} - {}".format(type(self).__name__,
		                                      type(other).__name__,
		                                      result))
		print(type(result))
		return result

# print( {} == {}) # things got bad here huh

class TestProxy(unittest.TestCase):
	""" test for base proxying of objects """

	def setUp(self):
		""" construct basic objects """
		self.baseDict = dict(makeTestObjects()["baseDict"])

	def test_baseProxy(self):
		""" test that tree objects find their root properly """
		p = Proxy(self.baseDict)
		self.assertEqual(self.baseDict, p, msg="""
		proxy object is not equal to base""")
		self.assertIsNot(self.baseDict, p, msg="""
		proxy object IS base""")
		self.assertIs(p._proxyObj, self.baseDict)

	def test_proxyTyping(self):
		""" test that object class is properly mimicked """
		p = Proxy(self.baseDict)
		self.assertIsInstance(p, Proxy, msg="Proxy is not a proxy instance")
		#print("proxy type", type(p), type(p).__bases__)
		self.assertIsInstance({}, type(p),
		    msg="dict is not an instance of generated proxy class")
		self.assertIsInstance(p, dict, msg="Proxy is not an instance of dict")

		class TempCheck(object):
			pass
		#print("isinstance", isinstance(p, TempCheck))
		self.assertFalse(isinstance(p, TempCheck),
		                 msg="Proxy is instance of any random thing")


	def test_proxySuper(self):
		pass

	def test_proxyList(self):
		baseList = ["test"]
		pList = DebugProxy(baseList)
		newList = ["new"]

		self.assertEqual(pList[0], "test")

		# both = newList + pList
		# print(both)

	def test_proxyRef(self):
		p = Proxy(self.baseDict)

		# add new key in baseDict
		self.baseDict["newKey"] = "newValue"
		
		# check new key retrieval
		self.assertEqual(p["newKey"], "newValue",
		    msg="proxy-base link retrieval is incorrect")

		# new key in proxy
		p["newProxyKey"] = "newProxyValue"
		self.assertEqual(self.baseDict["newProxyKey"], "newProxyValue",
		    msg="proxy-base insertion is incorrect ")






	def test_proxyEq(self):
		# p = Proxy(self.baseDict)
		p = DebugProxy(self.baseDict)
		self.assertEqual(p, self.baseDict)
		self.assertEqual(self.baseDict, p)

		# pp = Proxy(p)
		pp = DebugProxy(p)
		self.assertEqual(pp, self.baseDict)
		self.assertEqual(self.baseDict, pp)

		# I don't know how proxies should act with other proxies
		# self.assertEqual(pp, p)
		# self.assertEqual(p, pp)


class TestDeltaList(unittest.TestCase):

	def setUp(self):
		self.baseList = makeTestObjects()["list"]

	def test_baseListDelta(self):
		d = ListDeltaProxy(self.baseList)

		self.assertEqual(d, self.baseList)

		d.append("appItem")

		print(d._deltaStack)

		self.assertEqual(d, self.baseList + ["appItem"])
		self.assertFalse(d == self.baseList)

		self.baseList.insert(1, "insItem")

		self.assertEqual(d, ["a", "insItem", "b", "c", "d", "appItem"])

		self.assertEqual(d, d._product())

		print("")

		print(d._product())
		print(self.baseList)
		print(d, "t")
		print(d._product())

		d._baseObj = ["newBase"]

		print(d)




if __name__ == '__main__':
	unittest.main()



