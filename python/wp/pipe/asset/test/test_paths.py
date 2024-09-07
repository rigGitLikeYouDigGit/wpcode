
from __future__ import annotations

import os, shutil

import unittest

from wp import constant
from wp.constant import getAssetRoot, setTesting, TEST_ASSET_ROOT, ASSET_ROOT
from wp.pipe.asset.main import Asset
from wp.pipe.asset.bank import AssetBank


def clearTestAssetDir():
	"""clears the test asset directory"""
	assert getAssetRoot() == TEST_ASSET_ROOT, "Not testing, not clearing test asset dir, absolutely not"
	if getAssetRoot().exists():
		shutil.rmtree(getAssetRoot())
	os.mkdir(getAssetRoot())
	pass

# def makeAssets():
# 	"""create some test assets, save them, load them again"""
#
# 	caitAsset = Asset(tags={"character" : "cait"})
# 	caitBodyAsset = Asset(tags={"character" :"cait",
# 	                            "part" : "body"})
# 	benAsset = Asset(tags={"character" : "ben"})
# 	benBodyAsset = Asset(tags={"character" : "ben",
# 	                           "part" : "body"})
# 	benHeadAsset = Asset(tags={"character" : "ben",
# 	                           "part" : "head"})
# 	benTestFieldAsset = Asset(tags={"character": "ben",
# 	                           "part": "body",
# 	                                "use" : "render"})
#
# 	assets = [caitAsset, caitBodyAsset, benAsset, benBodyAsset, benHeadAsset,
# 	          benTestFieldAsset]
# 	return assets
#
# class Test_AssetBank(unittest.TestCase):
#
# 	def setUp(self) -> None:
# 		"""clear asset test dir, set up dummy assets"""
# 		setTesting(True)
# 		clearTestAssetDir()
#
# 		# if we're not testing, abort
# 		self.assertNotEqual(getAssetRoot(), ASSET_ROOT, "Test asset root points to production, aborting asset test")
# 		assert getAssetRoot() != ASSET_ROOT, "Asset root points to production assets, erroring out of asset test: " + __file__
# 		self.assertEqual(getAssetRoot(), TEST_ASSET_ROOT, "Not testing, aborting asset test")
# 		assert getAssetRoot() == TEST_ASSET_ROOT, "Not testing, erroring out of asset test: " + __file__
#
# 		self.assert_(getAssetRoot().exists(), "Test asset root not created")
#
#
# 		self.assertEqual(len(tuple(getAssetRoot().iterdir())), 0, "Test asset root not empty")
#
# 		self.assets = makeAssets()
#
# 		# test asset bank
#
# 		bank = AssetBank()
# 		for i in self.assets:
# 			bank.saveAssetToFile(i)
#
# 		bank.addAssets(self.assets)
# 		self.assertEqual(len(bank.assets), len(self.assets), "Not all assets added")
#
# 		self.bank = bank
#
# 	def testTags(self):
# 		"""test tags, filtering and retrieval"""
# 		bank = self.bank
#
# 		# get list of unique tag pairs to compare against
# 		tagPairs = set()
# 		for i in self.assets:
# 			tagPairs.update(i.tags.items())
#
# 		lex = tuple(bank.getSearchIndex().searcher().lexicon("tags"))
#
# 		self.assert_(len(lex) > 0, "No tags indexed")
#
# 		self.assertEqual(len(tagPairs), len(lex), "Unique tag pairs not equal")
#
#
#
# 		uidAsset = bank.assetFromUid(self.assets[0].uid)
# 		self.assertIs(uidAsset, self.assets[0], "Asset retrieved by uid not the same object")
#
# 		searchAssets = bank.searchAssetsByTags({"character" : "cait"})
# 		#print(searchAssets)
# 		self.assertEqual(len(searchAssets), 2, "No assets found by inclusive tag search")
#
# 		shouldFail = False
# 		try:
# 			exact = bank.searchAssetsByTags({"part": "body", "character" : "ben"}, exact=True)
# 		except KeyError:
# 			shouldFail = True
# 		if shouldFail:
# 			self.fail("Exact search failed")
#
# 		try:
# 			exact = bank.searchAssetsByTags({"part": "head", "character": "ben",
# 			                         "use": "render"}, exact=True)
# 		except KeyError:
# 			pass # correct behaviour
# 		else:
# 			self.fail("Invalid exact search did not raise KeyError")

#
if __name__ == '__main__':

	setTesting(True)
	print("testing", constant.TESTING)

	#clearTestAssetDir()
	#makeAssets()
