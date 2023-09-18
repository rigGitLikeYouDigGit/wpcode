
"""there's just something about nested structures that
pulls me in like an addict
"""


from .item import SuperItem
from .plugin import SuperItemPlugin

from .stlplugin import ListSuperItemPlugin, DictSuperItemPlugin

SuperItem.registerPlugin(ListSuperItemPlugin, (tuple, list))
SuperItem.registerPlugin(DictSuperItemPlugin, dict)