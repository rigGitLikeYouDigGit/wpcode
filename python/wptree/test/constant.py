
"""constant objects used across multiple tests"""
from __future__ import print_function
import pathlib

from tree import Tree

singleTree = Tree("testRoot")

threeTree = Tree("a")
threeTree("b", "c", create=True)

tempTree = Tree(name="testRoot", value="tree root")

branch = Tree(name="newBranch")

tempTree.addChild(branch)

#raise
tempTree("branchA", create=True)
#tempTree("branchA", create=True).value = "first branch"
#raise
tempTree("branchA")("leafA", create=True).value = "first leaf"
#raise
tempTree("branchA")("leafB", create=True).value = "2nd leaf"
#raise
tempTree("branchB", create=True).value = 2

#raise

midTree = tempTree.copy()
midTree("branchA")("listLeaf", create=True).value = ["a", "b", 10, ["2e4", False], "d"]
midTree("dictBranch", create=True).value = {"key" : "value", "oh" : {"baby" : 3}}
midTree.default = {}
#midTree("branchA").filePath = pathlib.Path(__file__)


if __name__ == '__main__':
    print("")
    print("####")
    print("")
    tempTree("branchA", "leafA").remove()
    print("")
    print([i.name for i in tempTree.allBranches()])

