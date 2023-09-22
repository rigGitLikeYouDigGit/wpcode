
"""there's just something about nested structures that
pulls me in like an addict
"""

from .base import (
	SuperItem,
	SuperModel,
	SuperViewBase,
	SuperDelegate,
)

from .stlplugin import (
	ListSuperItem,
	DictSuperItem,
	LiteralSuperItem,
	# StringSuperItem,
	# FloatSuperItem
)

for i in [ListSuperItem,
          DictSuperItem,
		  LiteralSuperItem,
		  #StringSuperItem,
		  #FloatSuperItem
          ]:
	#print(i, i.forCls)
	SuperItem.registerPlugin(i, i.forCls)