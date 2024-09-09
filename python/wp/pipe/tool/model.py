
from __future__ import annotations
import typing as T
from typing import TypedDict

from wptree import Tree
from wp.pipe import Show, Asset, AssetBank

from wpdex import WpDex, WpDexProxy
"""sketch for model tracking which show is used"""


def setModel(proxy:WpDexProxy):
	proxy.setRule("show", options=lambda : Show.availableShows())

# relying on Trees doesn't cover things like setting limits of min/max for numeric values
# in nested data structures
# BUT it would be so much simpler than the full wpdex proxy
def showModel(name="show", default="tempest"):
	t = Tree(name, value=default)
	t.auxProperties["options"] = lambda : Show.availableShows()
	t.auxProperties["ui"] = None
	return t


{"tool" : {
	"model" : {
		"rules" : {
			("*/leaf", ) : {"minMax" : (-1, None)},
			"/show" : {"options" : lambda : Show.availableShows()}
		} # list of wpDex rules to apply to values when set?
	}
}}

class MyTool(Tree):

	@classmethod
	def defaultBranchCls(cls):
		return Tree

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.addChild(showModel())
		#self["show"] = ShowModel()




def myTool():
	tool["mainShow"] = showModel(default="tempest")
	tool["assetBank"] = AssetBank(tool["mainShow"])
	tool["mainAsset"] = assetModel(allowRules=[HasMesh, HasArmature],
	                               bank=tool["assetBank"])


def runTool():

	print("myShow is ", tool["mainShow"].value, tool["mainAsset"].value)

if __name__ == '__main__':
	"""minimum test for me - make """

	print(Show.availableShows())
	from param import rx


