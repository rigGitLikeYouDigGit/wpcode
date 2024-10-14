
from __future__ import annotations

import shutil
import typing as T

from whoosh.fields import Schema, ID, KEYWORD, TEXT
from whoosh.index import Index
from whoosh.qparser import QueryParser, AndGroup
from whoosh import index as indexModule, writing

from wplib import log
from wp.pipe.asset import Asset, Show
from wp.constant import WP_ROOT, WP_LOCAL_ROOT

"""searching and filtering assets"""

SEARCH_DIR = WP_LOCAL_ROOT / "assetSearch"

#TODO: actually spend time on this, this is the absolute MVP

# build schema for searching assets
assetSchema = Schema(
	#uid=ID(stored=True),
	#tags=KEYWORD(stored=True),
	path=TEXT(stored=True)
	#dirPath=STORED()
	# tags=TEXT(stored=True,
	#           phrase=False,
	#           #analyzer=IDTokenizer(), # don't split tags
	#           ),
)

def getIndex()->Index:
	if SEARCH_DIR.exists():
		shutil.rmtree(SEARCH_DIR)
	if not SEARCH_DIR.exists():
		SEARCH_DIR.mkdir(parents=1, exist_ok=1)
		indexModule.create_in(str(SEARCH_DIR), schema=assetSchema)
		index = indexModule.open_dir(str(SEARCH_DIR))
		# syncIndex(index)
	index = indexModule.open_dir(str(SEARCH_DIR))
	# sync index all the time now
	# TODO: get this to run lazily on load or something
	syncIndex(index)
	return index

def syncIndex(index:Index):
	"""iterate over every asset and put it here
	for now we replace the original one
	"""
	log("syncIndex")
	writer = index.writer()
	for top in Asset.topAssets():
		for asset in top.allBranches(includeSelf=True):
			log("check asset", asset, isinstance(asset, Asset))

			if not isinstance(asset, Asset): continue
			writer.add_document(path=asset.strPath())
	writer.commit(#mergeType=writing.CLEAR
	              )

def searchPaths(path="", limit=10)->list[str]:
	"""todo: clean this, restructure this, everyone sing along"""
	index = getIndex()
	qp = QueryParser("path", schema=assetSchema)
	q = qp.parse(path)
	with index.searcher() as searcher:
		results = searcher.search(q, limit=limit)
	return [i["path"] for i in results]

def allPaths()->list[str]:
	index = getIndex()
	reader = index.reader()

	#with index.reader() as reader:
	result = reader.field_terms("path")
	#reader.close()
	return result




