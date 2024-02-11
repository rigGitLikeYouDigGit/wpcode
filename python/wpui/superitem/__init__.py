
"""there's just something about nested structures that
pulls me in like an addict


actually quite promising -
	hierarchical display for any arbitrary structure,
	allowing editing of any value, and adding/removing

for now it depends on the Qt item model and table view system -

for hierarchy we chain SuperItem -> SuperModel -> SuperItem -> SuperModel,
each level only knowing about its immediate children

later, consider each view defining vertical and horizontal modes
add option to collapse levels, use type labels to shorthand what
levels contain, etc

line widget for showing path to item, and for listing / collapsing

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
	NoneSuperItem,
	LiteralSuperItem,
	# StringSuperItem,
	# FloatSuperItem
)

