
from __future__ import annotations

import shutil
import typing as T

from whoosh.fields import *
from whoosh.analysis import *
from whoosh.spelling import *
from whoosh.query import *
from whoosh.index import Index
from whoosh.qparser import *
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
	#path=ID(stored=True, unique=True, sortable=True)
	path=KEYWORD(stored=True,
	             lowercase=True,
	             scorable=True,
	             analyzer=StandardAnalyzer()
	          #sortable=True,
	          #phrase=False,
	          #analyzer=IDTokenizer(),
	          #spelling=True,
	          #analyzer=StemmingAnalyzer()
	          )
	#dirPath=STORED()
	# tags=TEXT(stored=True,
	#           phrase=False,
	#           #analyzer=IDTokenizer(), # don't split tags
	#           ),
)




def getIndex()->Index:
	# if SEARCH_DIR.exists():
	# 	shutil.rmtree(SEARCH_DIR)
	if not SEARCH_DIR.exists():
		SEARCH_DIR.mkdir(parents=1, exist_ok=1)
		indexModule.create_in(str(SEARCH_DIR), schema=assetSchema)
		index = indexModule.open_dir(str(SEARCH_DIR))
		# syncIndex(index)
	index = indexModule.open_dir(str(SEARCH_DIR))
	# sync index all the time now
	# TODO: get this to run lazily on load or something
	# syncIndex(index)
	return index

def syncIndex(index:Index):
	"""iterate over every asset and put it here
	for now we replace the original one
	"""
	#log("syncIndex")
	writer = index.writer()
	for top in Asset.topAssets():
		for asset in top.allBranches(includeSelf=True):
			#log("check asset", asset, isinstance(asset, Asset))

			if not isinstance(asset, Asset): continue
			writer.add_document(#path=asset.strPath()
								path=asset.path
			                    )
	writer.commit(#mergeType=writing.CLEAR
	              )

def _startIndex():
	index = getIndex()
	syncIndex(index)
_startIndex()

class VeryFuzzyTermPlugin(FuzzyTermPlugin):
	"""remove the single-digit limit on how fuzzy a query can be"""


pathParser = QueryParser("path", schema=assetSchema,
                         termclass=FuzzyTerm
                         )
pathParser.add_plugin(FuzzyTermPlugin())

def searchPaths(path="", limit=10)->list[str]:
	"""todo: clean this, restructure this, everyone sing along"""
	index = getIndex()
	log("search", path, index.doc_count())
	log("all", allPaths())
	# qp = QueryParser("path", schema=assetSchema,
	#                  )
	qp = pathParser
	#q = qp.parse((path + "~9").strip()) # RIDICULOUSLY slow, this ain't it
	q = qp.parse(path)
	log("q", q)
	with index.searcher() as searcher:
		# newQ = searcher.correct_query(q, path)
		# results = tuple(searcher.search(newQ, limit=2))
		# corrector = Corrector().suggest()
		#results = searcher.search(q, limit=limit)
		results = tuple(searcher.search(q, limit=2))

		log("results", results)
		return [i["path"] for i in results]

def allPaths()->list[str]:
	index = getIndex()
	reader = index.reader()

	#with index.reader() as reader:
	result = reader.field_terms("path")
	#log("field terms", result)

	#result = reader.stored_fields("path")
	# result = reader.iter_field("path")
	#reader.close()
	return [i for i in result if "/" in i]

if __name__ == '__main__':
	log(allPaths())


