
from __future__ import annotations

import uuid

"""lib for uids and identifiers"""

import json, random
import typing as T

# from ..lib.path import Path
# from tree import ROOT_TREE_DIR
#
# WORD_POOL_PATH = ROOT_TREE_DIR / "resource" / "random_word.json"
# with WORD_POOL_PATH.open("r") as f:
# 	data = json.load(f)
# nWordsTotal = len(data["words"])
# assert nWordsTotal > 10, "need at least 10 words in word pool"
#
# # lambda to get uid of any compatible object
# toUid = lambda x: x if isinstance(x, str) else x.uid
#
# def getReadableUid(seed=None, nWords=3):
# 	"""return a sequence of random words from list -
# 	currently a 1 in 1 trillion chance of collision
# 	"""
# 	if seed is not None:
# 		random.seed(seed)
# 	#return "_".join(data["words"][random.randint(0, nWordsTotal-1)] for i in range(nWords)).strip()
# 	return str(uuid.uuid4())

def getUid4(seed=None, nWords=3):
	"""return a sequence of random words from list -
	currently a 1 in 1 trillion chance of collision
	"""
	if seed is not None:
		random.seed(seed)
	return str(uuid.uuid4())


if __name__ == '__main__':
	pass

	# print(len(data["words"]))
	# a = getReadableUid()
	# b = getReadableUid()
	# print(a == b)
	# seed = "test"
	# a = getReadableUid(seed)
	# b = getReadableUid(seed)
	# print(a == b)